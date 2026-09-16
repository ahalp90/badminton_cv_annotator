# Can the available evidence separate a court from a convincing false match?

## Evidence and revision actually accessed

**Repository:** `ahalp90/badminton_cv_annotator`
**Revision:** `7299ff3`, resolved to **`7299ff3f74bf51dc3bc59582c79190dbc81666ca`**. All repository reads were pinned to that revision, not the current head of `fix/court-det`. The supplement identifies `b90518c` as its scientific basis.

Paths below are relative to `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/`.

| Evidence                                                                                   | Actual access                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluation/README.md`, `automatic_axes_results.md`, `automatic_axes_visual_judgements.md` | Read completely, first and in the requested order.                                                                                                                                                                                                                                              |
| `evaluation/method_excerpts.md`                                                            | Read the court/mapping and relevant paint, visibility and winner-selection excerpts.                                                                                                                                                                                                            |
| `measurements.json.gz`, `gx0_control_measurements.json.gz`                                 | Fully recovered and decompressed; both compressed files verified against their Git blob hashes.                                                                                                                                                                                                 |
| `evaluation/ranking_records.json.gz`                                                       | **Partial retrieval:** recovered four complete detailed entries—Amateur-2 frame150 `30:30`, `30:33`, and frame28019 `16:1800`, `184:4123`, all `automatic_all_camera`. Their scores, gates and corners match the complete measurement summary. The remaining detailed entries were not decoded. |
| `automatic_axes_visual_check.html`, `gx0_control_visual_check.html`                        | Automatic gallery source partially read; GX0 control HTML/SVG source completely read. **No rendered gallery or background image was successfully opened.** Visual conclusions below use the existing written judgements, not my own image inspection.                                           |
| `methods.md`; commit metadata                                                              | Read as supplementary scope and provenance evidence.                                                                                                                                                                                                                                            |

## Assessment

**The observations expose missing discriminating information; they do not establish a court-identity or acceptance rule.** They refute the sufficiency of perfect paint profiles, complete profile availability, or the original floor outcome considered alone. They also expose useful disagreements between paint and line evidence—but do not establish that an untested combination resolves those disagreements without losing valid courts.

This is **not** a proof that the original images, or every combination of recorded features, cannot distinguish the candidates. The existing human judgements already distinguish several of them. It is a demonstration that the proposed interpretations of particular measurements are stronger than those measurements support.

## 1. A passed profile identifies a local appearance condition, not court paint

**Recorded facts.** All three identities below belong to population `automatic_all_camera`; the case remains part of each identity. Line values are `stripe.exclusive.score` in the detailed records and `stripe_score` in the compact summary.

| Exact case / candidate                   | Saved paint evidence                                                                         |   Line score | Existing judgement                                           |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- | -----------: | ------------------------------------------------------------ |
| `am2_window_01_frame_28019` / `184:4123` | `profile.score = 1.0`; all 12 intervals available and passing; all 11 marking values equal 1 | 0.1352180041 | Grossly wrong court near the net top                         |
| `shuttleset_03_scene_0019` / `165:6702`  | `profile.score = 1.0`; five available markings pass; six markings unavailable                | 0.2238074951 | Unrelated, hallucinated court                                |
| `am2_window_00_frame_150` / `30:33`      | `profile.score = 1.0`; all 12 intervals available and passing; all 11 marking values equal 1 | 0.3263949156 | Approved as perfect, with uncertainty at the extreme far end |

**Deduction:** the first and third candidates have identical complete saved `profile` objects but opposite visual rulings. A decision using only that object—including availability counts—cannot distinguish them. Their other features and images are not identical. Consequently, a visible-count penalty alone cannot explain or resolve `184:4123`.

For scene19 `165:6702`, the five supported markings are **left/right singles and doubles sidelines, and far short service**. The centre, both baselines, far long service, near short service and near long service are unavailable. That is five available intervals out of twelve, because the unavailable centre contains two intervals. It is **not six observed paint failures**.

### Three states—and the limits of their interpretation

| State                                             | What the records establish                                                                                                                               | What they do not establish                                                                                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Unavailable observation**                       | `interval_visible=False`; a marking is `None` when none of its intervals is eligible.                                                                    | Whether physical paint is absent, occluded, cropped away or distinguishable. An accompanying `interval_ridge=False` is an initialized placeholder, not a failed observation. |
| **Observed support**                              | On an eligible interval, the bright-ridge test passed. A marking value of 1 means its available intervals passed.                                        | That those pixels belong to this court, that one continuous marking supports the whole interval, or that another marking contributes independent evidence.                   |
| **Observed contradiction to the appearance test** | On an eligible interval, `interval_ridge=False` records a failed test. A centre value of 0.5 can combine one passing and one failing available interval. | That the court is physically wrong. Occlusion, weak contrast, crowded paint, resolution or insufficient valid side samples can also produce failure.                         |

These are **measurement states**. Semantic contradiction—expected paint is observably missing or replaced by incompatible structure—additionally requires knowing that the relevant region is observable and that the expected marking should be distinguishable there. The arrays do not supply that knowledge. Conversely, the written rejection of `184:4123` supplies semantic evidence against the candidate even though every profile passes.

`detector._visible_samples` requires a positive clipped span of **at least 12 working pixels**. An intersecting but shorter interval is unavailable too. Homography, image dimensions and model intervals can distinguish geometric clipping reasons; they cannot establish occlusion or contrast. `inspect_appearance.profiles` averages available intervals within each marking, then averages non-`None` markings. Observing just one centre-line half can therefore supply a full centre-marking value.

### How projection can remove inconvenient evidence

**Deduction from the scoring method:** let \(A(H)\) be the markings eligible under candidate homography \(H\). The recorded score is

$$
P(H)=\operatorname{mean}\{r_m(H):m\in A(H)\}.
$$

Candidate geometry determines both where observations are tested and which markings enter the denominator. Holding the remaining outcomes fixed, removing outcomes below the current mean increases that mean. Across candidates, projecting poorly aligned markings outside the image—or leaving less than 12 working pixels—can therefore remove negative contributions rather than incur contradiction.

This is a **selection incentive**, not a claim about an optimizer’s intention. Changing an actual homography can also change the remaining profiles, camera eligibility and line support; improvement is not guaranteed.

Scene19 establishes a perfect **conditional** score with limited observations. It does not establish what the six unobserved markings would have scored outside the image. Counting them as failures invents evidence; counting them as successes does likewise. A valid severely cropped court could have the same availability pattern. The detailed records accessed here do not establish an approved five-marking analogue with which to validate a completeness penalty.

**Untested explanation for the all-eleven case:** `_filter_painted_stripes` samples 24 positions, searches five normal offsets at each position and accepts an interval when at least 40% satisfy its centre-versus-both-sides contrast condition. Those are existing measurement settings, not proposed acceptance thresholds. Passing every profile does not certify continuous paint at every position or a coherent offset along each line. Fragmented or repeated background ridges could satisfy the predicate, but the records do not verify which structures supplied `184:4123`’s passing samples. Eleven passes cannot be treated as eleven independent confirmations.

## 2. Existing corroboration is informative, but its simplest vetoes have contrary examples

**Recorded facts:** both false paint winners have `gates.floor_score = -1.0`. So do the approved **GX0 supplied-direction comparator `89`** and **GX0 label-guided observed-bank winners `1864` and `5144`**, judged essentially ideal. Their common case is `gxBQ_window_00_frame_0`, but their populations differ from the automatic winners. The floor outcome alone therefore does not separate these false and approved candidates. Nothing supplied establishes which internal floor condition produced `-1.0`.

The false winners also have `geometry_valid=True`, `player_fractions=[1.0,1.0]`, and camera errors **0.0244304924** and **0.0694164338**, respectively. Approved Amateur-2 frame150 `30:33` also has player fractions `[1.0,1.0]`. Those existing player summaries do not resolve the ambiguity, and camera eligibility is not an identity certificate.

There is a more specific discrepancy already worth examining. In `184:4123`, `stripe.exclusive_per_marking` is exactly zero for **left singles, centre and right doubles**, although all their paint profiles pass; `gates.line_counts` is `[0,2]`. Scene19 `165:6702` has `[0,0]`. But demanding strong exclusive support on every marking faces a positive contrary example: approved `30:33` has far-long-service exclusive support of approximately **2.25e-15**, while its paint profile passes. These measurement systems are not interchangeable tests of physical absence.

The following are **bounded hypotheses about additional discrimination**, not proposed gates:

| Observation and named ambiguity                                                                                | What it could distinguish; physical assumption required                                                                                                                                                                                                                                                                                                          | Plausible valid court it would wrongly penalise                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Finite extent, endpoints and centre-line gap:** Amateur-2 mat-border fits; partial scene19 pattern           | Support should occupy the appropriate finite spans and, where observable, terminate or meet other markings consistently. Current profiles already use finite clipped intervals; merely adding “finite extent” is not a new distinction. Stronger termination evidence requires observable endpoints and assumes unrelated markings do not continue through them. | Cropped endpoints, worn or player-covered terminations, or intersecting multi-sport paint. A blanket in-frame rule has actual contrary examples: GX0 controls `1864` and `5144` have a native corner at `y=1161.095…` in a 1080-high image; approved `30:33` has a corner at `x=-98.425…`. These establish valid partial extent, not validity of every low-profile-count crop. |
| **Cross-marking spatial correspondence using line/profile disagreement:** `184:4123`’s eleven passes           | Could distinguish separately corroborated markings and junctions from borrowed or fragmented structures. Requires resolvable paint identities and correspondence between saved assignments and image locations. A common homography imposes model consistency; it does not independently verify observed-support identities.                                     | Foreshortened, blurred or crowded far-end markings can merge; genuine junctions share pixels. Approved `30:33` already contradicts uniform strong exclusive support. A field named `independent` does not establish statistical independence.                                                                                                                                  |
| **Player-to-floor/court association beyond saved fractions:** GX5 wall/children fits and Amateur-2 net-top fit | Reliable foot/contact positions associated with actual players could locate the playing surface relative to the candidate. Requires correct player identity, treatment of jumping/occlusion and an assumption about where players may stand.                                                                                                                     | Players lunging outside the court, jumping, partly occluded or confused with nearby people. Requiring every detected person inside the court would be unjustified.                                                                                                                                                                                                             |
| **Scene support distinguishing floor paint from wall, net or mat edge:** GX5 and Amateur-2 false fits          | Could distinguish physical surface or object identity when ridge appearance is similar. Requires trustworthy scene/plane association; a mat boundary must not be assumed to be a court boundary. This is not equivalent to reinstating the old floor gate.                                                                                                       | Valid low-contrast, reflective, heavily marked or partly hidden flooring. More decisively, the recorded gate already rejects the approved GX0 trio. Its missing implementation prevents attributing those outcomes to a particular mechanism.                                                                                                                                  |

The first two can be partly investigated through existing geometry and stripe arrays. Verifying spatial ownership, meaningful absence or scene identity requires image evidence and often explicit visual judgement. I did not obtain that image access; aggregate fields cannot replace it.

## 3. These are opposing ranking counterexamples, not an acceptance evaluation

**Recorded facts:** the required choice changes with case and population. Errors below are the recorded maximum corner distances at **1280 × 720**, not acceptance thresholds.

| Population and case                          | Existing comparison                                                                                                                                                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `automatic_all_camera`, Amateur-2 frame150   | Line `30:30` follows the mat boundary; paint `30:33` is approved. Errors: **185.106 versus 10.843**.                                                                                |
| `automatic_all_camera`, ShuttleSet03 scene19 | Line `1:60` is usable; paint `165:6702` is rejected. Errors: **12.736 versus 8390.859**.                                                                                            |
| `automatic_all_camera`, GX0                  | Neither `22:4579` nor `22:4588` is usable. The user prefers line despite its larger error: **18.806 versus 12.690**.                                                                |
| Camera-first arm, Amateur-2 frame28019       | Paint `16:1796` has error **22.900**; removing the global cap introduces all-camera paint winner `184:4123` at **1189.522**. The smaller error is not visual approval of `16:1796`. |

**Deduction:** choosing line everywhere loses approved `30:33`; choosing paint everywhere loses usable scene19 `1:60`. Replacing the all-eleven false winner with its line winner `16:1800` would still leave an unusable Amateur-2 frame28019 result. Selecting an alternative and accepting it are different questions.

Perfect paint agreement is not necessary for the existing usability rulings either: the essentially perfect Amateur-3 line winner `(am3_window_00_frame_0, automatic_all_camera, 43:22603)` has paint score **0.7272727…**. Scene17 line winner `1:80` is usable, whereas paint winner `1:132` is described as worse with a far-boundary overshoot. No overall approval was recorded for that paint panel; it should not silently become a newly labelled binary negative. Accepted paint-edge and internal-line differences elsewhere do not define a universal tolerance. Apparent bowing has no verified cause.

| Claim level                | What follows here                                                                                                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Candidate availability** | GX0 controls establish matcher capacity with label-guided observed directions, not automatic recovery. Ranking cannot recover candidates excluded earlier. Poor displayed nearest-control diagnostics do not prove every undisplayed candidate unusable.     |
| **Ranking**                | `run_automatic.winner_ids` compares camera-eligible candidates with an available paint score. Line maximises exclusive line support; paint maximises profile score with line support as tie-breaker. Floor outcome does not select these diagnostic winners. |
| **Acceptance**             | Human approval is candidate-specific. Winner status does not mean the detector accepts or emits that candidate; no acceptance rule is established here.                                                                                                      |
| **End-to-end detection**   | Five views have usable line winners and four have usable paint winners. Their six-view union requires human choice between rankings, not an automatic six-of-nine success rate.                                                                              |

These distinctions are explicit in the methods and judgements. The **19,286** all-camera candidates are not 19,286 labelled negatives in the export: it contains **17 distinct automatic winners plus two GX0 controls**. Nine views from five videos—including ShuttleSet median composites—are development material, not independent trials or designated holdouts. Score calibration and population false-positive rates cannot be inferred. The two additional hard videos remain reserved as specified.

## One smallest fixed-pool comparison: observability and correspondence

**Specification only—not run.** Use the **19 distinct exported winners**: both rankings across all nine automatic views, deduplicating the shared scene20 winner, plus GX0 observed-bank controls `1864` and `5144`. This is the smallest already-exported set retaining both rankings on every development view and both ideal observed-bank contrary controls. Keep supplied-direction `89` as a documented comparator, not another automatic candidate or independent success. Never merge scene16 and scene19 `1:60`.

**Question.** Where perfect paint agreement is wrong, is corroboration missing or spatially incompatible—information hidden by aggregation—or does the observable background genuinely supply a coherent finite pattern, leaving court identity unresolved? These explanations can coexist. The comparison must allow **indeterminate**.

**Required features.** Freeze candidates, scores and existing judgements. Compare per marking: `interval_visible`, `interval_ridge`, `marking_ridge`, `stripe.exclusive_per_marking`, assignments, gates, homography, image dimensions and painted-interval mapping. Retrieve the remaining detailed ranking entries before interpreting their unexamined fields. Use existing backgrounds to assess observable support/contradiction and whether support belongs to distinct, correctly connected finite markings. This last step needs spatially interpretable provenance or explicit visual judgement; aggregate assignment arrays alone are insufficient.

**Positive contrary examples and recorded outcomes.** Preserve approved Amateur-2 `30:33` against mat-border `30:30`; usable scene19 `1:60` against false `165:6702`; and false frame28019 `184:4123` despite full profile availability. Preserve approved GX0 controls despite floor rejection/cropping, and Amateur-3, scene16 and scene20 positives despite acceptable marking discrepancies. Retain scene17’s missing overall paint approval rather than inventing a label.

**Discriminating outcomes and failure conditions.** Observable extent/junction or line/profile correspondence contradictions on the false candidates, absent from the positive contrary examples, would support a specific additional-evidence hypothesis **within this set**. A false candidate remaining coherent under those checks would show that those checks have not resolved identity. A cue that also condemns approved partial or weakly marked courts fails as a sufficient discriminator. Inaccessible images, unresolved observability, unknown provenance or indistinguishable evidence states make the comparison indeterminate. No off-image paint outcome may be invented to rescue it.

This remains a **development diagnostic**, not tuned-and-tested generalisation, a full-pool ranking replay or an acceptance benchmark. It does not alter the independent direction experiments or their fixed protocol.

## What can be concluded without more annotation

The identical-profile counterexample, conditional-mean missingness mechanism, distinction between unavailable and failed tests, floor/player counterexamples, opposing ranking preferences and separation of evaluation stages follow from frozen methods, recovered measurements and existing judgements. No new direction result or pixel cutoff is needed.

What remains unresolved is **which physical structures supplied the false winners’ passing samples**, whether a failed test means observable missing paint rather than weak or occluded paint, and whether finite/joint correspondence separates the contrary examples. Those require image-level judgement or evidence beyond the accessed summaries. Diagnosing the floor gate requires its implementation; evaluating its consequences requires appropriately judged rejected-candidate evidence. Completing my partial archive retrieval would fill an access gap, not automatically supply these semantic labels.

**Bottom line:** “this candidate explains the sampled appearance” and “the observations identify this as the court” remain different claims. The evidence establishes that gap more strongly than it establishes a replacement rule.

### Handover

[Full Markdown assessment](sandbox:/mnt/data/court_identity_review/handover/assessment.md) · [Candidate evidence ledger](sandbox:/mnt/data/court_identity_review/handover/candidate_evidence_ledger.json) · [Complete handover bundle](sandbox:/mnt/data/court_ranking_review_7299ff3.zip)

The bundle includes the access manifest, four retrieved detailed ranking entries and both verified compact source archives. It contains no duplicated gallery images.

**Yes. I completed a read-only numerical audit that adds two substantive findings:** an explicit court-length ambiguity in the scene19 false winner, and a specific marking alias in the all-eleven Amateur-2 false winner. These narrow the failure mechanisms considerably, but neither supplies a validated acceptance rule.

## Evidence and revision actually accessed

Everything remains pinned to **`7299ff3f74bf51dc3bc59582c79190dbc81666ca`** in `ahalp90/badminton_cv_annotator`, not the current branch head.

I re-verified the complete `measurements.json.gz` and `gx0_control_measurements.json.gz` against their committed Git blob hashes. I used the four previously recovered complete detailed ranking entries, cross-checking their scores, gates and corners against the verified compact measurements. Newly read source includes `net_geometry.py`, `stripe_observations.py`, `paint_geometry.py`, relevant `detector.py` and `assignment.py` sections, and `player_guided.py`. The guide and relevant method excerpts were also re-read.

**The detailed ranking archive remains only partially recovered.** I obtained the scene19 PNG header, which establishes its native dimensions as **960 × 540**, but did not open its pixels or render the galleries. There are no new visual judgements in this extension.

[Full Markdown report](sandbox:/mnt/data/court_extension/handover/extension_report.md) · [Numerical audit record](sandbox:/mnt/data/court_extension/handover/results/audit_results.json) · [Reproducible handover bundle](sandbox:/mnt/data/court_identifiability_extension_7299ff3.zip)

## 1. Scene19’s available pattern leaves a specific court-length degree of freedom

**Candidate:** `(shuttleset_03_scene_0019, automatic_all_camera, 165:6702)`.

The recorded starting point is familiar: perfect paint score, five available markings, six unavailable, and a gross visual rejection. What is new is that the available geometry admits an explicit family of different court extents—not merely a favourable missing-data denominator.

### The clipping audit identifies what is actually missing

Reconstructing the homography from the verified saved corners and replaying the documented clipping calculation reproduces **all twelve saved availability flags**.

For this candidate, the seven unavailable intervals—the centre contributes two—have **no image intersection**. None is an intersecting interval that merely falls below the existing 12-working-pixel sampling requirement.

| Available interval     | Clipped length, working pixels | What its visible ends represent                        |
| ---------------------- | -----------------------------: | ------------------------------------------------------ |
| Left doubles sideline  |                     959.032629 | Image clipping boundaries                              |
| Left singles sideline  |                     959.032629 | Image clipping boundaries                              |
| Right singles sideline |                     959.032629 | Image clipping boundaries                              |
| Right doubles sideline |                     959.032629 | Image clipping boundaries                              |
| Far short service      |                     101.320419 | One clipping boundary and one projected model endpoint |

Thus, **none of the four available longitudinal markings supplies an observed baseline termination**. That is a geometric statement about the sampled model intervals, not an assertion that physical paint is absent or occluded. The interval-by-interval calculation is retained in the [audit record](sandbox:/mnt/data/court_extension/handover/results/audit_results.json); its clipping semantics follow `_visible_samples`.

### A transformation preserves every available paint-sampling interval

Let \(H\) be the saved candidate’s court-to-image homography and \(y_0=4.72\) metres, the far short-service coordinate. Define

$$
T_a=
\begin{bmatrix}
1&0&0\\
0&a&(1-a)y_0\\
0&0&1
\end{bmatrix},
\qquad H_a=HT_a,\qquad a>0.
$$

This changes longitudinal scale about the far short-service line. It leaves every constant-\(x\) supporting line unchanged as a line, and fixes \(y=y_0\) pointwise. While the sideline ends remain beyond the image, their clipped intervals therefore remain unchanged too.

For the fixed analytical examples **\(a=0.95,\ 1.00,\ 1.05\)**, I verified:

* The same twelve-entry availability mask.
* The same five available clipped intervals, with maximum endpoint discrepancy of **\(9.1\times10^{-13}\)** working pixels.
* Nevertheless, a maximum corner displacement of **397.739439 working pixels** from the saved false court at either non-unit scale.

These transformed homographies are **mathematical witnesses, not claimed members of the generated candidate pool**. The displacement is relative to the saved false candidate, not an error against the true court.

The transformation also preserves this candidate’s two vanishing directions: its first homography column is unchanged and its second is only rescaled. The exhibited ambiguity concerns placement and extent conditional on those directions; it is not measured by a change in their directions.

### The existing camera-error bound leaves this ambiguity intact

I replayed the committed `_camera_from_corners` calculation, including its float32/OpenCV reconstruction and focal grid. It reproduces the saved camera error **exactly**:

`0.06941643378323906`.

At the original best grid focal length—approximately **3.521953 image widths**—the camera objective along this transformation is

$$
E(a)=\sqrt{c^2+(r-\log a)^2},
$$

where \(c\) is the calibrated-axis cosine and \(r\) the log axis-norm ratio at that fixed focal length. Rescaling the second axis leaves the cosine unchanged and subtracts \(\log a\) from the norm ratio.

| Analytical scale \(a\) | Maximum corner change, working pixels | Camera error at the **unchanged focal length** |
| ---------------------- | ------------------------------------: | ---------------------------------------------: |
| 0.95                   |                            397.739439 |                                       0.086753 |
| 1.00                   |                                     0 |                                       0.069416 |
| 1.05                   |                            397.739439 |                                       0.084418 |

All three remain below the **existing 0.1 bound**, without choosing a new focal length. At this focal alone, the inequality allows approximately \(0.931238\leq a\leq1.075441\). This is a derived compatibility interval, not a proposed acceptance threshold.

The float64 reconstruction used for the geometric proof differs from the source’s float32 camera reconstruction by approximately \(9.96\times10^{-9}\) in the baseline objective; both calculations are recorded separately. I did **not** rerun pixel profiles or claim bitwise identity of their outputs: boundary rounding can matter even when geometric coordinates agree to numerical precision.

Nor does this establish full detector eligibility for the witnesses. Player evidence, floor outcomes, net-depth checks and generation membership were not replayed. Passing this approximate camera objective is not proof of an exact physical calibration.

### What observation would actually constrain the missing dimension?

The ideal named-line constraint matrix for the five supporting lines has **rank 7**, leaving one homography degree of freedom after homogeneous scale. Adding the centre supporting line alone leaves rank 7. Adding a second distinct transverse marking raises it to rank 8.

That is an algebraic information calculation, **not an assumption of independent marking noise**.

A correctly identified second transverse marking, or a genuine longitudinal endpoint at a known court coordinate different from \(y_0\), can break this particular ambiguity. Another image-crop endpoint cannot. Even the available endpoint on far short service does not break the exhibited mode, because that line is fixed pointwise.

The necessary physical assumption is that the observation belongs to the named court marking, and that any apparent termination is physical rather than caused by cropping, weak paint or an occluder. **A real cropped court showing the same five markings would be equally underdetermined by these observations.** Consequently, this is an observability warning, not a rule for rejecting partial courts.

**What this adds:** scene19 is now an explicit example of *unchanged available paint-sampling geometry plus acceptable camera-error objective failing to identify court extent*. A visible-count penalty would not supply the missing dimension.

## 2. The all-eleven Amateur-2 winner contains a measurable marking alias

**Candidate:** `(am2_window_01_frame_28019, automatic_all_camera, 184:4123)`.

This requires a different explanation: all twelve intervals are available and pass, giving eleven marking scores of one. The missing-dimension argument above cannot explain that complete profile count.

### Two nominal markings become subpixel-separated

The saved homography places the clipped **near baseline only 0.378426–0.452247 working pixels from the near long-service supporting line** across the baseline’s in-image span.

These are projected nominal line locations, not a new measurement of actual painted edges. Their separation is much smaller than the existing profile’s normal-centre search offsets, `[-4, -2, 0, 2, 4]` working pixels. The profile test does not enforce exclusive ownership of support across markings.

That identifies a concrete opportunity for two expected markings to use almost the same image neighbourhood. Without the pixels, it does **not** establish which ridge actually made either binary profile pass.

### The saved assignments independently expose the competition

The detailed `stripe` fields show:

| Marking           | `independent_per_marking` | `exclusive_per_marking` |
| ----------------- | ------------------------: | ----------------------: |
| Near baseline     |              0.8077183990 |  \(6.27\times10^{-19}\) |
| Near long service |              0.8350072396 |            0.8244762207 |
| Centre            |              0.2453800454 |                       0 |

Two exact entries of `stripe.assignments` are particularly informative:

| Zero-based assignment-array position | Chosen marking / position         | Chosen reverse strength | Best alternative marking | Alternative reverse strength |
| -----------------------------------: | --------------------------------- | ----------------------: | ------------------------ | ---------------------------: |
|                                  133 | Near long service / negative edge |            0.9971234116 | Near baseline            |                 0.9733605228 |
|                                  174 | Near long service / negative edge |            0.8040646855 | Near baseline            |                 0.6967659000 |

These are **assignment-array positions, not asserted raw-fragment IDs**. Their image coordinates and physical object identities were not recovered. The strengths are descriptive responses, not probabilities. The exact values and derived comparisons are in the [numerical record](sandbox:/mnt/data/court_extension/handover/results/audit_results.json).

The newly inspected source makes the interpretation precise. `resolve_fragments` gives each fragment one marking/position pair by maximum mean **reverse** support. `score_model` subsequently restricts forward coverage to fragments assigned to that identity. The near baseline’s high nonexclusive coverage therefore does not survive the recorded assignment, while near long service retains high coverage.

**Important limitation:** this assignment optimises the separable reverse objective, not joint forward coverage. The result demonstrates competition under the existing assignment convention; it does not prove that every possible exclusive assignment must lose the near baseline. It also does not prove these fragments supplied the binary pixel-profile passes.

### The approved contrary example prevents a simple exclusivity veto

The approved Amateur-2 frame150 paint winner, `(am2_window_00_frame_150, automatic_all_camera, 30:33)`, has the same complete all-one profile object. Yet its far long-service forward support also drops—from **0.1979269199** to **\(2.25\times10^{-15}\)**—under the recorded exclusive assignment. Its predicted far long-service/baseline separation is **1.861547–2.243938 working pixels**.

The aggregate loss from independent to exclusive forward support is:

| Exact case / candidate, all `automatic_all_camera`                | Relative forward-support loss |
| ----------------------------------------------------------------- | ----------------------------: |
| Amateur-2 frame150 / `30:30` — rejected mat-boundary fit          |                         7.47% |
| Amateur-2 frame150 / `30:33` — approved paint fit                 |                         5.20% |
| Amateur-2 frame28019 / `16:1800` — rejected line fit              |                         7.29% |
| Amateur-2 frame28019 / `184:4123` — rejected all-eleven paint fit |                    **36.13%** |

These are descriptions of existing fields, **not a tuned replacement score**. The unusually large loss strengthens the evidence-reuse diagnosis for `184:4123`; the other rejected fits show that the loss does not provide a general separator.

Requiring every marking to retain exclusive support would reject approved `30:33`. A hard minimum projected separation would also risk rejecting a valid strongly foreshortened court. The existing examples do not establish a universal resolution cutoff.

**What this adds:** “all eleven profiles pass” can conceal both subpixel proximity between different expected markings and explicit competition for fragment identity. The relevant distinction is **availability versus separately attributable support**, not merely availability versus missingness.

## What the completed comparison establishes

The targeted geometry audit covers **nine exact candidates**: the four recovered Amateur-2 winners, scene19’s line/paint pair, GX0 observed-bank controls `1864` and `5144`, and supplied-direction comparator `89`. The last three remain separate control/comparator populations. All 17 distinct compact automatic winners were retained unchanged in the handover, but this is not a full audit of the detailed 19-entry export.

| Candidate group                                                            | Available intervals in geometry replay | Ideal supporting-line rank |
| -------------------------------------------------------------------------- | -------------------------------------: | -------------------------: |
| Scene19 false `165:6702`                                                   |                                      5 |                          7 |
| Scene19 usable `1:60`                                                      |                                     12 |                          8 |
| Four Amateur-2 candidates, including approved `30:33` and false `184:4123` |                                12 each |                     8 each |
| GX0 controls `1864`, `5144`, and comparator `89`                           |                                12 each |                     8 each |

The rank-8 result is conditional on correct named-line correspondence. It is **not** a quality certificate: several rejected Amateur-2 fits have it. Conversely, the approved GX0 controls retain their original floor rejection while having complete interval availability and full ideal line rank.

No more annotations were needed to establish the clipping pattern, mathematical ambiguity, camera-objective compatibility, projected line spacing or saved assignment competition. What remains unavailable is the **pixel-level ownership evidence** needed to identify which structures supplied the passing profiles, and the observability judgement needed to turn a failed appearance test into a claim of physically absent paint.

I also inspected the legacy scorer, but did not establish the complete archived wrapper/settings-to-export trace or recover all rejected candidates. I therefore do not assign a new cause to the floor failures.

These are development diagnostics, not generalisation results. They neither assume independent frames nor convert descriptive scores into probabilities. Analytical witnesses are not generated alternatives; no winner was replaced; no acceptance or end-to-end detection improvement is claimed. The independent direction protocol and reserved hard videos remain untouched.

## Handover

The [complete bundle](sandbox:/mnt/data/court_identifiability_extension_7299ff3.zip) contains the [Markdown report](sandbox:/mnt/data/court_extension/handover/extension_report.md), frozen inputs with their access limitations, offline audit and plotting scripts, [numerical results](sandbox:/mnt/data/court_extension/handover/results/audit_results.json), provenance, and the analytical figure in [PNG](sandbox:/mnt/data/court_extension/handover/results/scene19_camera_ambiguity.png) and [SVG](sandbox:/mnt/data/court_extension/handover/results/scene19_camera_ambiguity.svg). A fresh offline rerun produced byte-identical JSON result files.

**The meaningful advance is a separation of two failure mechanisms: scene19 lacks an observed constraint on extent; Amateur-2 has complete nominal coverage but evidence that marking support is not distinct. Addressing one does not establish that the other has been resolved.**
