# WebUI red-team review: corrected contact and refit experiment

**Date:** 10 August 2026  
**Scope:** Review of `plan.md`, `findings.md`, `decisions.md`, and the public repository code named by those documents.  
**Repository inspected:** `ahalp90/badminton_cv_annotator` public `main` as available during this review.

## Executive verdict

| Experiment | Verdict | Reason |
|---|---|---|
| Experiment 1: incoming motion at the earliest accepted contact | **FAIL as written** | The core inference is no longer circular, but the qualifying-path rule, GT anchor taxonomy, no-path semantics, and headline threshold-selection rule are not specified tightly enough to make the result reproducible and audit-safe. These are small corrections, not a redesign. |
| Experiment 2: prepend and rerun `fit_alternation` | **PASS** | Prepending `None` cleanly isolates the contact-count parity change, and prepending the inferred other player is a legitimate second counterfactual. The latter must be described as an **additional Experiment-1-derived vote**, not statistically independent evidence, because its player identity is the deterministic complement of the anchor attribution already present in the sequence. |

There is **no repeat of the previous circular mistake in the main design**. The plan explicitly anchors on the earliest accepted contact, obtains that contact's player from `attribute_half`, and forbids the old alternating-fit label from choosing the approached player; repository inspection confirms that `attribute_half` itself does not consult `fit_alternation`, `fitted_first_all`, or GT.

Experiment 1 still needs a few exact rules before implementation. Once those are fixed, I would change its verdict to PASS without asking for a different model, new annotations, or production refactor.

---

## What each experiment actually tests

### Experiment 1 — two sentences

Experiment 1 tests whether the shuttle's **pre-contact** trajectory closes on the player directly attributed by `attribute_half` at the earliest accepted contact strongly enough to classify that contact as a first return rather than a serve. A positive detection assigns the other player as the contact-local server; this must be kept separate from the old alternating-fit baseline and from cases where no qualifying motion evidence exists.

### Experiment 2 — two sentences

Experiment 2 tests what happens to the existing alternating fit when exactly one missing position is inserted before the accepted-contact sequence on Experiment-1-positive rallies. Prepending `None` changes only sequence parity, while prepending the inferred other player changes parity and adds one extra vote derived from Experiment 1 before calling `fit_alternation` once.

---

# Blocking faults in Experiment 1

## 1. The lookback horizon and path-selection rule are not defined

`plan.md` says to "look backwards from the anchor for a continuous court-view shuttle path" and gives minimum length and maximum end-gap rules, but it never states **how far back to search** or exactly which run is selected if more than one run qualifies.

That is not a cosmetic omission. The decisions document already identifies the unresolved choice: use a **30 base-30-frame lookback** to match the research question, rather than the production serve rule's 25-frame lookback. Public code confirms that the production constant is currently `serve_start_lookback_frames=25.0` in `src/annotator/fps_constants.py::scale_for_fps`, so silently reusing that constant would change the experiment.

### Smallest correction

Before implementation, define the path window explicitly:

- scale **30 base-30 frames** with the repository's frame-count scaling rule, if that pending recommendation is approved;
- search only in `[anchor - lookback_frames, anchor)`;
- use the **closest-to-anchor maximal consecutive run** satisfying the path evidence rules;
- require its final visible point to be no more than two scaled base-30 frames before the anchor;
- never search further back merely because an earlier run looks cleaner.

Do not reuse the production 25-frame constant unless that is explicitly chosen instead.

---

## 2. "Stays in court view" is not enough to exclude camera-cut joins

The plan requires court view, recurrence code zero, consecutive visibility, minimum movement, and a gross-jump cap. Those checks handle gaps, stationary paths, recurrence artefacts, and many single-frame hallucinations, but they do **not explicitly prevent a path from crossing a scene/homography boundary**.

The repository already has the correct simple seam: `src/annotator/rally/evidence.py::tracker_segments` builds maximal intervals by intersecting each homography scene row with court-present runs. That is stronger than checking `court_present` alone. A camera cut can leave court detection true on both sides, and a jump-ratio guard should not be asked to double as a scene-cut detector.

### Smallest correction

Require the complete qualifying path and the anchor to lie in the **same `tracker_segments` interval / homography scene row**. Keep the jump guard as a trajectory-quality check, not as the camera-cut guard.

This is a one-rule correction and directly answers the red-team requirement about camera-cut joins.

---

## 3. "Unambiguous match to contact 1 or 2" needs an operational definition

The plan correctly says that GT is used only for evaluation and exploratory threshold choice, and that later, ambiguous, and unmatched anchors are reported separately. However, it does not define what makes an anchor "unambiguous".

That matters because the planning findings already observed an anchor within tolerance of both contacts 1 and 2, and several GT first/second pairs close enough that tolerance ambiguity is real rather than theoretical.

The public repository gives the right building blocks:

- `src/annotator/calibration/scoring.py::classify_all` maps a GT rally to a predicted span only when every GT stroke lies in exactly one same span;
- `src/annotator/calibration/gt_scoring.py::canonical_tolerance` scales the canonical five-base-30-frame contact tolerance;
- `src/classifier_shared/player_mapping.py::collect_shots` exposes `ball_round`, frame, and mapped Top/Bottom player information.

### Smallest correction

For a **covered** GT rally with one mapped predicted span, classify the earliest accepted anchor against GT strokes using the canonical tolerance:

1. zero GT strokes within tolerance -> `unmatched`;
2. more than one GT stroke within tolerance -> `ambiguous`;
3. exactly one, `ball_round == 1` -> `contact_1`;
4. exactly one, `ball_round == 2` -> `contact_2`;
5. exactly one, `ball_round >= 3` -> `later`.

Only strict `contact_1` and `contact_2` rows enter the binary first-return threshold selection/evaluation. Do not use GT server identity, the alternating phase, or the labels of other accepted candidates to choose the anchor player or trajectory feature.

---

## 4. No qualifying path is currently allowed to masquerade as evidence that the anchor was the serve

`plan.md` currently says: if the return rule fires, infer the other player as server; **otherwise name the anchor player as server**, while merely marking rallies with no qualifying motion path.

That creates an interpretation problem. "No qualifying path" means the motion hypothesis was not measurable; it is not positive evidence that the earliest accepted contact was the serve. The decisions document already proposes the right safeguard: show both a forced anchor-player attribution and an evidence-only abstention.

### Smallest correction

Report two explicitly different contact-local server variants:

- **forced coverage variant:** return trigger -> other player; otherwise -> anchor player;
- **evidence-only variant:** return trigger -> other player; **no qualifying path -> abstain**. If a qualifying path exists but does not pass the return threshold, the row may be reported as a negative detector decision, but keep path coverage explicit.

For the binary contact-1/contact-2 detector, report both path coverage and end-to-end precision/recall/F1 so missing paths cannot disappear from recall. For server attribution, count evidence-only no-path rows as abstentions.

This keeps "absence of motion evidence" separate from "evidence of a serve".

---

## 5. The headline operating-point selection rule is not predeclared

The plan promises a "selected threshold" and plots precision, recall, and F1, but it does not say **how the selected threshold is chosen**. With several swept path-quality cut-offs plus a direction threshold on the same three videos, choosing the prettiest point after seeing final server accuracy would make the headline result researcher-dependent.

Using the same three videos is acceptable for this stated EDA, but only as an **exploratory/in-sample** result and only if the operating-point rule is fixed before inspecting the outcome.

### Smallest correction

Predeclare one main threshold-selection rule on the strict GT `contact_1` versus `contact_2` subset. A defensible default is:

> choose the threshold that maximises first-return F1; break exact ties by higher precision, then by the stricter return threshold.

If "strongly enough" is intended to mean a precision-constrained detector instead, choose that precision floor **before running the sweep**. In either case, final server macro-F1 may be shown as a secondary curve but must not select the main threshold.

Do not jointly optimise every path-quality sweep and the classifier threshold against the same headline metric without labelling that result as exploratory sensitivity.

---

# Checks that pass

## No circular anchor-player choice

**PASS.** `src/annotator/point_winner.py::attribute_half` checks shuttle visibility, reads the sticky per-slot distances, chooses the nearest finite sticky slot, and maps that picked player's bbox foot position above/below the net band. It does not call or read `fit_alternation`, `_phase_assignment`, `fitted_first_all`, or GT.

The exact-tie behaviour is worth recording: NumPy's first minimum resolves a two-slot distance tie to Top. That is a diagnostic edge case, not a blocking fault.

## "Earliest accepted" is a real pipeline concept

**PASS.** `src/annotator/rally/contacts.py::impulse_cell_candidates` returns candidate frames in ascending order; `assemble_contacts` retains raw candidates with wrist-gate and suppression status. `src/annotator/video_outcomes.py::scoring_filter` keeps candidates unless `wrist_near is False` or `suppressed is True`, then `build_contact_data` removes the definitive exclusion mask and appends frames to `filtered_by_rally` in preserved order.

Therefore the first frame in a rally's filtered list is the earliest accepted contact under the current pipeline. Earlier raw candidates can be useful diagnostics, but the plan is correct not to silently promote a failed/suppressed/excluded impulse into a credible contact.

## Quadratic fit is in the right place

**PASS.** The direction-only detector is named as primary, and the quadratic is explicitly a diagnostic/alternate comparison. Keep it that way; lower quadratic residual alone is not evidence of a physically real parabola.

## Same-video threshold tuning is acceptable for this EDA

**PASS with labelling.** There is no train/test split, so no threshold-selected number should be described as held-out, validated, or generalisable. For a small student-project EDA whose purpose is to test whether the signal is promising, full curves plus an explicitly exploratory operating point are acceptable.

## Evaluation categories are mostly separated correctly

**PASS after the no-path correction above.** The plan already separates the old alternating-fit baseline, the contact-local estimate, the prepend/refit result, first/second/later/ambiguous/unmatched anchors, the all-rally denominator, covered rallies, and the frozen known-failure subset.

Keep every denominator written as `numerator / denominator`, and count unknown anchor halves and evidence-only no-path cases as abstentions rather than silently removing them.

---

# Experiment 2: direct parity and vote check

## Verdict: PASS

The design is mechanically legitimate and does not need a fabricated serve frame.

`src/annotator/point_winner.py::fit_alternation` consumes only an ordered list of `Top`, `Bot`, or `None` guesses. It scores the two possible alternating phases and returns the fitted **final** half; `src/annotator/video_outcomes.py::_first_stroke_half` derives the first half from that final half and the sequence length.

## Why prepending `None` is a clean parity-only experiment

For an original sequence of length `n`, original contact `j` is assigned relative to the final index `n-1`. After prepending one element, that same contact moves to index `j+1` while the final index becomes `n`, so its parity distance is unchanged:

`n - (j + 1) == (n - 1) - j`.

Therefore every original contact keeps the same phase assignment and contributes exactly the same vote. The prepended `None` contributes no vote, so:

> `fit_alternation([None] + guesses) == fit_alternation(guesses)`

for the experiment's sequences, including ties.

What changes is the interpretation of the **first** player because the sequence length increases by one. This is exactly the missing-contact parity effect the plan says it wants to isolate.

The dedicated sequence tests should assert this identity directly.

## Why the player-labelled prepend is legitimate, but not an independent player measurement

When Experiment 1 says the anchor is a first return, the proposed server half is `OTHER_HALF[anchor_half]`. Adding that half before the accepted guesses supplies one additional vote to the alternating phase consistent with the anchor.

That is a valid counterfactual, but the phrase "one independent player vote" is too strong. The server identity is a deterministic complement of the same direct anchor attribution that is already present as the first accepted-contact guess; motion supplies the **decision to insert a missing serve**, not a separately measured server identity.

Use this wording instead:

> "prepend the inferred other player, which tests the parity change plus one additional Experiment-1-derived vote that is independent of the existing alternating fit, but not independent of the anchor attribution."

A useful expected-behaviour check follows from the one-vote structure: the new vote can strengthen the anchor-consistent phase, resolve an existing tie, or turn a one-vote opposing win into a tie. By itself it cannot jump directly from one resolved final-half winner to the opposite resolved winner in a single added vote.

## Do not oversell Experiment 2

The player-labelled variant partly **reweights the anchor attribution**. It does not demonstrate a second independent observation of who served.

That does not invalidate the experiment. It just means the parity-only result is the clean isolation of "one contact was missing", while the player-labelled result is the effect of additionally trusting the Experiment-1 player implication.

---

# Non-blocking risks

1. **Recurrence code zero is not proof of a real shuttle.** `src/annotator/inpaint_guard.py` grades exact recurrence patterns; code zero means "no recurrence flag", not "verified genuine". Keep the producer-original-only sensitivity proposed in the findings if the provenance sidecar is available.

2. **Too many jointly tuned knobs can overfit 16-ish strict positive examples.** The preliminary findings report a very small strict contact-2 group. Regenerate the counts, show the full sensitivity surface/curves, and avoid choosing movement floor, jump cap, net closure, and direction percentage solely by whichever combination produces the highest final metric.

3. **`attribute_half` has a deterministic Top tie bias.** Record exact two-slot distance ties as a diagnostic count.

4. **All-GT-rally and covered-rally denominators must not be conflated.** `classify_all` gives no unique mapped span for split/missed GT rallies. In an all-GT-rally table, those rows need an explicit no-anchor/not-covered state rather than disappearing.

5. **The preliminary counts in `findings.md` are not results.** The document itself says the 292/249/87/17/etc. counts must be regenerated and checked. Do that before copying any number into the final report.

6. **A helper cited in the planning findings was not present on public `main`.** `src/annotator/calibration/serve_prepend_measurement.py::run_contact_injection_counterfactual` could not be found in the public branch inspected here. Experiment 2 does not need it: the public `point_winner.fit_alternation` seam is sufficient.

7. **The proposed third commit message is slightly misleading.** "Prepend an order-only server guess" conflates the `None` parity-only slot with the labelled server variant. Prefer wording that names both the missing-contact slot and the inferred-player sensitivity.

---

# Smallest concrete corrections to the plan

Make only these changes before implementation:

1. **Path window:** state the lookback horizon, scale it explicitly, and define the closest-to-anchor maximal qualifying run.
2. **Cut guard:** require anchor and path to remain inside one `tracker_segments` / homography-scene interval.
3. **GT taxonomy:** define unique-within-canonical-tolerance contact 1, contact 2, later, ambiguous, and unmatched classes exactly.
4. **No-path semantics:** show both forced-anchor and evidence-only-abstaining server attribution; never describe no motion evidence as evidence of a serve.
5. **Threshold rule:** predeclare the main first-return operating-point rule on strict contact-1/contact-2 truth; keep server macro-F1 secondary.
6. **Experiment-2 wording:** replace "independent player vote" with "additional Experiment-1-derived vote, independent of the alternating fit but not of the anchor attribution."
7. **Sequence tests:** explicitly test `fit_alternation([None] + guesses) == fit_alternation(guesses)` and the expected one-vote effects of the labelled prepend.

No production code change, neural model, new annotations, fake serve frame, or wider architecture work is needed.

---

# Questions that must be answered before implementation

## 1. Which lookback is the experiment actually using?

The decisions document recommends **30 base-30 frames** because that matches the original research question; production currently uses 25. This must be settled before the trajectory extractor is written.

**Reviewer recommendation:** use 30 for this EDA and scale it with the repository's normal base-30 frame-count rule. Do not change the production constant.

## 2. What exactly selects the headline first-return threshold?

Choose one rule before looking at the final curves:

- maximum strict contact-1/contact-2 first-return F1; or
- a predeclared precision floor with maximum recall underneath it.

**Reviewer recommendation if no stronger precision requirement exists:** maximise first-return F1, tie-break by higher precision and then the stricter threshold. Do not select the operating point by final server macro-F1.

Those are the only questions I consider genuinely blocking. The remaining corrections can be written directly into the plan.

---

# Final red-team disposition

**Experiment 1: FAIL as written, narrowly.** It asks the correct non-circular question and uses the correct direct anchor player, but it is not yet auditably reproducible because the lookback/run choice, scene-boundary rule, GT ambiguity rule, no-path interpretation, and threshold-selection policy are incomplete.

**Experiment 2: PASS.** The parity-only prepend is exactly described and has a clean algebraic interpretation; the player-labelled prepend is also legitimate, provided it is reported as an additional Experiment-1-derived vote rather than a statistically independent measurement.

After the small corrections above, the two experiments answer the intended research question without using `fitted_first_all`, the alternating phase, or GT server identity to choose the anchor player or calculate incoming motion.

---

## Repository evidence inspected

The following public functions/files were checked directly:

- `src/annotator/point_winner.py`
  - `attribute_half`
  - `_phase_assignment`
  - `fit_alternation`
- `src/annotator/video_outcomes.py`
  - `scoring_filter`
  - `build_contact_data`
  - `_first_stroke_half`
- `src/annotator/rally/contacts.py`
  - `impulse_cell_candidates`
  - `assemble_contacts`
- `src/annotator/rally/evidence.py`
  - `tracker_segments`
  - sticky-player evidence construction
- `src/annotator/inpaint_guard.py`
  - recurrence grading and code meanings
- `src/annotator/fps_constants.py`
  - `ScalingKind.FRAME_COUNT`
  - `scale_for_fps` and the shipped 25-base-30 serve lookback
- `src/annotator/calibration/scoring.py`
  - `classify_all`
  - `classify_rally_boundary`
  - canonical five-base-30 contact tolerance constant
  - `greedy_match`
- `src/annotator/calibration/gt_scoring.py`
  - `canonical_tolerance`
  - fixture/scene-row loading
- `src/classifier_shared/player_mapping.py`
  - `collect_shots`

The public implementation supports the central planning claims about direct anchor attribution, accepted-contact ordering, and alternating-fit parity. The remaining FAIL items above are specification/reporting gaps in Experiment 1, not evidence of a hidden dependency on the old fitted server label.
