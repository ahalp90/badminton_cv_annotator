# Feedback on the serve-start trajectory investigation report

The investigation appears useful, but the current report is very difficult to interpret because it repeatedly changes denominators and conditioning sets without first explaining the evaluation funnel. Several category labels are also too vague to recover what was actually measured.

Please revise the writeup so that a reader can understand **what population each number refers to, why rows are excluded at each stage, and exactly what each plotted category means** without having to inspect `metrics.json.gz` or the analysis code.

## 1. Start with an explicit evaluation funnel

The report should begin with a compact accounting table like:

| Stage | Count | Meaning |
|---|---:|---|
| GT rallies | 292 | All ShuttleSet GT rallies in the evaluated videos |
| Segmentation-covered GT rallies | 249 | Every GT stroke lies inside exactly one predicted rally span |
| Split GT rallies | 24 | GT strokes span multiple predicted spans and/or fall partly outside |
| Missed GT rallies | 19 | No GT stroke falls inside a predicted rally span |
| Clear first/second-contact anchors | 103 | Earliest accepted predicted contact uniquely aligns to GT stroke 1 or 2 |
| Usable motion paths among clear anchors | 19 | Clear anchors with a pre-contact shuttle path passing all path-quality gates |
| Motion-rule return calls among clear anchors | 14 | 11 TP + 3 FP at the selected recurrence-clean rule |
| Usable motion paths among all 249 covered rallies | 24 | Needed to interpret the later “motion answer only” result |
| Motion triggers among all 249 covered rallies | 16 | Used by the downstream server-attribution experiment |

The current writeup makes numbers such as **292, 249, 103, 19, 24, 14, and 16** appear almost interchangeably. They are not interchangeable and should never be shown without their denominator and conditioning rule.

## 2. Correct the explanation of 249 vs 292

Please do **not** describe 249 as “the number of rallies detected by the pipeline.”

There are 292 GT rallies, while the frozen pipeline emits more predicted rally spans than that. The 249 figure specifically means:

> 249 GT rallies are classified as `COVERED`: every GT stroke for that rally lies inside exactly one and the same predicted rally span.

The remaining GT rallies are:

- 24 `SPLIT`
- 19 `MISSED`

This distinction is important because the experiment is conditioned on rally segmentation being usable, not simply on whether some predicted rally exists.

Please state this explicitly near the top of the report.

## 3. Surface the merged-span issue

There is another important caveat currently missing from the report: the 249 covered GT-rally rows do not correspond one-to-one with 249 distinct predicted spans.

There are **244 distinct predicted spans** behind those 249 covered rows. Five predicted spans each contain two GT rallies, so ten GT-rally rows are being evaluated using only five underlying predicted spans.

Because `classify_all` can mark both GT rallies as `COVERED`, the same predicted contact sequence / anchor can effectively be evaluated against two GT rallies.

Please:

1. document this;
2. report the count of merged spans;
3. either exclude merged spans from the main contact/server analysis or show a sensitivity result with them removed.

The conclusion does not seem to change dramatically when excluding them, but this is a genuine evaluation-accounting issue and should not be hidden.

## 4. Explain the 103-anchor subset precisely

The phrase currently used for the 103 examples is too vague.

Please show the full earliest-anchor alignment breakdown among the 249 covered GT rallies:

| Earliest accepted predicted contact aligns to… | Count |
|---|---:|
| GT stroke 1 / serve | 87 |
| GT stroke 2 / first return | 16 |
| Later GT stroke | 3 |
| Ambiguous | 1 |
| No GT stroke within tolerance | 142 |
| Total | 249 |

Therefore:

> 103 = 87 unique first-contact matches + 16 unique second-contact matches.

This is **not** a limitation of ShuttleSet truth. It is a limitation of how often the pipeline's earliest accepted predicted contact can be uniquely aligned to GT stroke 1 or 2.

The fact that **142/249 anchors are unmatched** is a major result and should be prominent.


### 4a. Make clear that the anchor is not serve-gated

Please explicitly document how the “earliest accepted contact” is chosen.

As currently implemented, this anchor is **not required to look like a serve**. It comes from the ordinary contact chain and may qualify on the basis of:

- shuttle-trajectory impulse;
- the wrist-near body-unit gate;
- contact suppression / de-duplication;
- definitive exclusion masking.

The generic player-proximity flag is annotation/diagnostic information rather than the effective acceptance gate.

There is no requirement at this point for:

- serve-like shuttle stillness before launch;
- a serve-specific trajectory shape;
- service-court geometry;
- serving posture;
- rally-opening semantics;
- any other explicit “looks like a serve” criterion.

This matters because the investigation is currently treating the earliest ordinary accepted contact as an **anchor candidate**, then asking whether it is the serve or first return.

The analysis therefore needs to distinguish at least these three cases:

1. earliest accepted contact is the true **serve**;
2. earliest accepted contact is the true **first return** because the serve contact was missed;
3. earliest accepted contact is a **spurious ordinary contact candidate** that does not correspond to any GT stroke.

The third category must not be conflated with the “missed serve, therefore first return” hypothesis.

Please make that distinction explicit in the report and in the interpretation of the 142 unmatched anchors.

### 4b. Re-run anchor/GT alignment at ±5, ±10, and ±30 base-30fps frames

Please do **not** present the ±5 base-30fps join as the only meaningful alignment result.

For this project, the intended interpretation is:

- **±5 base-30fps frames** — ideal / strict timing agreement;
- **±10 base-30fps frames** — the normal **usable baseline** for contact evaluation;
- **±30 base-30fps frames** — a broad **sanity-check limit** for asking whether a candidate is plausibly the same physical stroke at all.

Please report the full earliest-anchor classification at **all three tolerances**:

| Join tolerance | Purpose |
|---|---|
| ±5 base-30fps | ideal / strict |
| ±10 base-30fps | usable baseline |
| ±30 base-30fps | sanity check |

For each tolerance, report at minimum:

- unique match to GT stroke 1;
- unique match to GT stroke 2;
- unique match to a later GT stroke;
- ambiguous match to multiple GT strokes;
- unmatched;
- total;
- unmatched fraction;
- ambiguous fraction.

Also report the distribution of the anchor's **nearest GT-stroke signed offset** and **absolute offset**, including at least median and useful quantiles.

The key diagnostic question is:

> Does the large unmatched population mostly disappear at the normal ±10-frame baseline, or do many anchors remain so far from any GT stroke that they are better interpreted as false contact candidates?

The ±30 sanity-check view is particularly important here. If an “earliest accepted contact” is still unmatched even within ±30 base-30fps frames, that is strong evidence that this is not merely strict timing disagreement.

Please **do not spend report space on ±2 frames**. That is tighter than the useful evaluation regime here and is even inside the expected region of manual GT annotation error. It is not a decision-relevant operating point for this investigation.

Where possible, show the tolerance sensitivity as a compact table rather than forcing the reader to infer it from separate metrics.

## 5. Rename the TrackNet / inpaint comparison categories

The plot titled something like **“does excluding TrackNet's filled or interpolated points help?”** is currently under-specified.

The two categories should be named according to what the code actually does.

Suggested labels:

- **Recurrence guard only**
  - keep otherwise-valid shuttle points only when the recurrence-pattern guard gives `NO_FLAG`
- **Recurrence guard + producer inpaint mask**
  - additionally exclude all points marked as inpaint-selected by the TrackNet/InpaintNet producer metadata

The first arm is not merely “exclude obvious hallucinations”: it excludes every non-`NO_FLAG` recurrence grade, including fabricated, suspect-flat, and degraded frames.

Please explain this in the caption or methods text.

## 6. Make the TrackNet comparison a controlled ablation

The current plot asks whether excluding producer-inpainted points helps, but the two variants appear to be allowed to choose different movement thresholds.

That confounds:

1. changing the path mask, and
2. changing the classifier threshold.

For the ablation figure, please hold the motion threshold fixed and vary **only** whether producer-inpainted points are excluded.

You can still report separately that each arm's best exploratory threshold differs, but the main “does this mask help?” comparison should change one thing at a time.

The high-level result appears to be:

### Recurrence guard only

- usable paths: 19 / 103
- TP: 11
- FP: 3
- FN: 5
- precision: 78.6%
- recall: 68.8%
- F1: 0.733

### Recurrence guard + producer inpaint exclusion

- usable paths: 12 / 103
- TP: 9
- FP: 0
- FN: 7
- precision: 100%
- recall: 56.2%
- F1: 0.720

The interpretation should therefore be something like:

> Excluding all producer-marked inpaint points removes the observed false positives but reduces usable-path coverage and loses true positives. It trades recall for precision rather than being an unambiguous improvement.

## 7. Explain the jump from 14 to 16 motion triggers

Among the 103 clear first/second-contact anchors, the selected motion rule produces:

- 11 TP
- 3 FP
- therefore 14 positive return calls.

Later, the server experiment refers to **16 motion triggers**.

Please state explicitly that the rule is then applied to **all 249 covered rallies**, not just the 103 cleanly labelled first/second-contact anchors, and it fires on two additional rallies outside the clear-anchor subset.

Without that sentence, 14 vs 16 looks like an inconsistency.

## 8. Clarify every server-attribution method in plain English

Please define the server-attribution methods before plotting them.

For example:

- **Existing alternating fit**  
  Current pipeline behaviour: fit Top/Bot alternation across accepted contacts.

- **Anchor player**  
  Assume the player associated with the earliest accepted predicted contact is the server.

- **Anchor player, flipped on incoming motion**  
  Use the anchor player normally, but if the pre-contact shuttle motion says the anchor is actually receiving a return, infer the other player as server.

- **Motion evidence only**  
  Same inference, but abstain unless a usable pre-contact motion path exists.

- **Prepend unknown player, then refit**  
  On a motion trigger, insert a missing pre-anchor contact without a player identity and rerun the alternation fit.

- **Prepend inferred other player, then refit**  
  Insert the inferred missing server before the accepted-contact sequence and rerun the alternation fit.

The current short labels require too much reverse engineering.

## 9. Keep segmentation coverage separate from server-attribution accuracy

The “All 292” and “Covered 249” bars are confusing because the normal methods appear to have exactly the same number of correct predictions in both cases.

That means the all-292 score is effectively:

> server-attribution result on covered rallies × rally-segmentation coverage.

Please do not present that as though it were a second server-attribution benchmark.

Instead, report:

> Rally segmentation usable for downstream evaluation: 249 / 292 = 85.3%.

Then use **249 covered rallies** as the main denominator for server-attribution accuracy.

If an end-to-end 292-rally metric is retained, label it explicitly as **end-to-end accuracy including segmentation failures**, not simply “All”.

## 10. Explain why the 121-rally subset gives the old method 0%

The report also evaluates 121 rallies where the released fit was wrong or missing.

Please say explicitly that this subset is **defined by failure of the existing method**, e.g.:

- 99 wrong
- 22 abstained
- 121 total

Therefore the old method necessarily scores 0/121 on that subset.

This is a useful **failure-recovery analysis**, but it is not another general benchmark and should be labelled accordingly.

## 11. Distinguish “path unavailable” from “classifier says serve”

This is especially important for interpreting the 103-anchor first-return experiment.

Only 19/103 clear anchors have a usable path.

A return counted as a false negative may therefore mean either:

- there was a valid path and the motion rule classified it as a serve; or
- no valid path existed, so the system could not produce positive motion evidence.

Please separate these cases where possible.

A confusion matrix alone hides the more important engineering question: **is the motion classifier wrong, or is motion evidence simply unavailable?**


### 11a. Diagnose the unmatched earliest-contact population before tuning the motion rule further

Before treating the serve-vs-return trajectory rule as the main bottleneck, please analyse the earliest accepted contacts that fail GT alignment.

For the unmatched population at each of the requested ±5 / ±10 / ±30 base-30fps tolerances, break down:

- how far the anchor is from the nearest GT stroke;
- whether it lies before or after that stroke;
- whether it occurs before the GT serve;
- whether it passed because of a strong impulse, wrist-near evidence, or both;
- whether suppression chose it over a nearby candidate that aligned better to GT;
- whether it falls in a merged predicted-rally span;
- whether TrackNet/inpaint artefacts or recurrence-guard states are implicated.

The objective is to distinguish:

> “contact detector timing is somewhat noisy”

from

> “the earliest accepted ordinary contact candidate is frequently not a real stroke.”

That distinction is more fundamental than fine-tuning the incoming-motion classifier.

## 12. Please make the report self-contained

The final report should let someone answer the following without opening code or compressed metrics:

- Why 292?
- Why 249?
- What happened to the other 43?
- Why 103?
- What happened to the other 146 covered rallies?
- Why 19?
- Why later 24?
- Why 14 return calls here but 16 motion triggers later?
- What exactly differs between the two TrackNet/inpaint arms?
- What is the denominator of every percentage?
- Are merged predicted rally spans included?
- Is an “all 292” server result measuring server attribution or also segmentation failure?

Please put denominators directly in plot labels/captions whenever possible, e.g. `11/16 returns`, `19/103 usable paths`, `164/249 covered rallies`, rather than percentages alone.

## Suggested replacement opening

Something close to the following would make the rest of the report much easier to follow:

> We evaluate 292 ShuttleSet ground-truth rallies. The frozen rally-segmentation output fully covers 249 of them: all GT strokes fall inside one predicted span. Another 24 GT rallies are split across predicted spans and 19 are missed, so downstream contact and server analyses use the 249 covered GT rallies unless otherwise stated.
>
> For each covered rally we take the pipeline's earliest accepted contact as the anchor. That anchor uniquely aligns to ShuttleSet stroke 1 in 87 rallies and stroke 2 in 16, giving 103 rallies where we can directly test whether pre-contact shuttle motion distinguishes a serve from a first return. The remaining covered anchors consist of 142 unmatched contacts, three later-stroke matches, and one ambiguous match.
>
> Of those 103 clear anchors, 19 have a pre-contact shuttle path passing all fixed path-quality gates. The selected incoming-motion rule identifies 11 of 16 first returns and falsely calls 3 of 87 serves as returns. Applied to all 249 covered rallies, 24 have usable paths and the rule fires on 16.
>
> For server attribution, assuming that the anchor player's contact is the serve scores 154/249 covered rallies. Flipping to the other player when incoming-motion evidence indicates that the anchor is a first return raises this to 164/249. More elaborate attempts to prepend a missing contact and rerun the old alternation fit do not help, scoring only 129–130/249.

## Requested outcome

Please revise the report primarily for **evaluation accounting and interpretability**, not cosmetic prose.

The main requirements are:

1. one explicit denominator/funnel section at the beginning;
2. precise names for every conditioning subset;
3. complete accounting of excluded/unmatched rows;
4. explicit documentation that the earliest anchor is an ordinary accepted contact, not a serve-gated detection;
5. anchor-to-GT alignment sensitivity at **±5, ±10, and ±30 base-30fps frames**, with ±10 treated as the usable baseline and ±30 as the sanity check;
6. clear definitions of every plot category and server method;
7. a controlled inpaint-mask ablation;
8. explicit treatment of merged predicted rally spans;
9. separation of segmentation failures, false/spurious contact anchors, evidence unavailability, and classifier errors;
10. captions that show counts as well as percentages.

The underlying investigation may be useful, but the current report makes it unnecessarily hard to determine what was actually evaluated and what conclusions the numbers support.
