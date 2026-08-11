# Decisions

## Deviations and open questions

- The progressive-disclosure extension is presentation-only and awaits approval of its fourth commit. It keeps the approximately 800-word summary required by the earlier audit, but adds the newly requested 120–180-word Bottom line before it. This resolves the only tension between the audit records.
- No open scope decisions block launch. The correction uses the user-approved population, alignment, diagnostic and terminology rules below.
- The planning records remain in this investigation directory because this is an extension of the existing pass. Starting a second worklog elsewhere would split the audit trail.
- Serena/Pyrefly now indexes only this dedicated investigation directory in addition to `src/**` and `tests`. The tracked project profile still excludes other tracked Python under `scratch/swarm_review/**`.
- The original 0.25-body-height cut-offs have no independent physical or calibration provenance. They first appear with the implemented analysis, while the pre-implementation plan said the cut-offs would be swept. Preserve them as historical comparisons rather than treating them as fixed physical facts.
- The older “Agreed”, “Choices used in the run” and “User approval” sections below record the completed original run. This correction section supersedes their population, contact-join and threshold-selection choices.

## Correction extension agreed on 11 August 2026

### Evaluation populations

**Decision:** use 239 one-to-one mapped rallies for analyses that assume one predicted span and contact sequence per GT rally. Keep 249 covered rallies as a sensitivity result. Use all 292 GT rallies only for a clearly labelled end-to-end view that includes segmentation failures.

Current flow:

```text
292 GT rallies
  -> classify_all(...)
  -> 249 COVERED rows
  -> anchor, motion and server scoring over all 249 rows
```

The current flow allows two GT rallies in one predicted span to reuse one accepted-contact sequence and anchor.

Corrected flow:

```text
292 GT rallies
  -> 249 COVERED rows + 24 SPLIT + 19 MISSED
  -> group COVERED rows by (fixture, predicted span)
       -> one GT rally in span: 239 one-to-one rows [PRIMARY]
       -> two GT rallies in span: 10 rows from 5 merged spans [EXCLUDED FROM PRIMARY]
  -> 249 COVERED rows [SENSITIVITY]
  -> 292 GT rallies [END-TO-END, INCLUDES SEGMENTATION FAILURE]
```

The correction must verify every count from the frozen inputs. Stop if the observed mapping differs from this approved structure.

### Contact alignment

**Decision:** rebuild the primary contact labels at ±10 base-30fps frames. Report ±5 as the strict sensitivity and ±30 as the physical-stroke sanity check.

Current flow:

```text
anchor frame + canonical_tolerance(fps)
  -> contact_1 | contact_2 | later | ambiguous | unmatched
  -> one stored label drives threshold selection and motion scoring
```

Corrected flow for each tolerance:

```text
anchor frame + every ordered GT stroke frame
  -> nearest GT ordinal
  -> signed offset = anchor - GT frame, in base-30fps frames
  -> absolute offset
  -> number of GT strokes inside the tolerance
  -> tolerance label: contact_1 | contact_2 | later | unmatched
  -> separate multiple-within-tolerance flag
```

The nearest ordinal and offset survive even when several strokes lie within ±30. The multiple-match flag records ambiguity without discarding the sanity-check information.

### Accepted-contact sequence after an unmatched anchor

**Decision:** for every primary-set anchor unmatched at ±10, inspect the remaining accepted contacts against all GT strokes at the same tolerance.

```text
accepted contacts ordered by frame
  -> match each contact independently to all GT strokes; matches do not consume a stroke
  -> retain its nearest GT stroke and flag multiple matches
  -> record whether a later contact matches the GT serve
  -> otherwise record whether a later contact matches the first return
  -> record the one-based accepted-contact rank of the first GT match in the full sequence
```

This separates a GT-incompatible early anchor followed by a recovered serve from a missing serve followed by a detected first return. Cases with no accepted contact matching either stroke remain a separate group.

### Predeclared motion-rule comparison

**Decision:** compare the historical absolute-closure rule with one predeclared 0.05-BH robust-trend rule. Do not select either rule or any threshold from the corrected classification scores.

The historical rule remains exactly:

```text
path eligibility:
  at least 5 sampled points
  end no more than 2 base-30fps frames before the anchor
  recurrence guard gives NO_FLAG
  finite contact-player distance
  total shuttle movement >= 0.25 apparent player body heights
  largest step / median non-zero step <= 4.0

return decision:
  net distance closure >= 0.25 apparent player body heights
  at least 55% of consecutive distance changes move towards the player
```

Both 0.25 values were introduced by the original analysis. The 55% threshold was selected on that analysis's old GT join. None has independent physical provenance.

The alternative uses the same path eligibility except for the absolute 0.25-body-height total-movement floor. For one path with contact-player distances `d[i]`, normalise sample time to `t[i] = i / (n - 1)` and calculate:

```text
robust_slope = median((d[j] - d[i]) / (t[j] - t[i])) for every i < j
intercept = median(d[i] - robust_slope * t[i])
residual[i] = d[i] - (intercept + robust_slope * t[i])
fitted_decrease = -robust_slope
residual_rms = sqrt(mean(residual[i] ** 2))
trend_to_jitter = fitted_decrease / residual_rms
```

The alternative calls the path incoming only when `fitted_decrease >= 0.05` apparent player body heights across the observed path. The `0.05 BH` value is an engineering judgement fixed before corrected scoring. It is not a calibrated physical constant and must not be swept or retuned.

`residual_rms` and `trend_to_jitter` are diagnostics only. When `residual_rms == 0`, record positive infinity for a positive fitted decrease, negative infinity for a negative decrease and zero for no decrease. Do not use either diagnostic to accept or reject a path.

The slope is a median of pairwise slopes so one bad endpoint cannot create the trend by itself. Residual RMS keeps remaining distance scatter visible rather than requiring every frame to approach. Both terms use the same apparent body-height distance scale, so the ratio is invariant to a constant rescaling. Frame-varying pose scale, player wrist motion and genuine curved approach can still contribute to the residual and must remain limitations.

Score both rules on the 239 one-to-one rallies with unique ±10 contact-1/contact-2 truth. Report full counts and per-video results. The purpose is to compare what the rules require, not to select the higher score.

For the 0.05-BH rule, report fitted decrease, residual RMS and trend-to-jitter as continuous measurements:

- GT serves versus first returns;
- correct versus incorrect calls;
- against the number of sampled path points;
- in representative false positives and false negatives.

Any diagnostic pattern is a finding only. Do not promote it into another classifier in this investigation.

For the inpaint ablation, apply both frozen rules unchanged to both masks:

```text
historical rule x recurrence guard only
historical rule x recurrence guard + producer inpaint mask
0.05-BH robust-trend rule x recurrence guard only
0.05-BH robust-trend rule x recurrence guard + producer inpaint mask
```

Use the 0.05-BH robust-trend rule for the main inpaint explanation because it was predeclared from the physical question. Keep the historical rule's fixed-mask result beside it as sensitivity. Do not retune either rule for either mask.

### Language for unmatched anchors

**Decision:** use `GT-incompatible anchor at the ±30 sanity limit` or `candidate unmatched within ±30` for unverified cases. Use `spurious contact` or `false contact` only for cases confirmed by read-only visual inspection, and report how many cases were inspected.

### Reporting and review

- Break the main funnel, alignment, path-availability, motion and server counts out by `sset_01`, `sset_15` and `sset_21`, as well as globally.
- Separate segmentation failure, anchor/GT mismatch, unavailable motion path, measured serve/return classification and server-attribution correctness.
- Make `FINAL_REPORT_READABILITY_AUDIT.md` and `PLOT_READABILITY_AUDIT.md` blocking final-output checks.
- Apply `write-clearly` for structure and self-containment, then `de-yuck` for plain human technical prose.
- Preserve the feedback and both readability audits unchanged as tracked review records.
- Use one WebUI fresh-reader stop after the complete report and supporting plots exist. The first read receives the report, the original questions and the plots, but not the plan, decisions, worklog or implementation notes.

## Agreed

- Anchor each rally at its earliest accepted geometry/impulse contact. The contact does not need to meet serve criteria.
- Determine the anchor player directly from contact geometry with `attribute_half`.
- Never use `fitted_first_all` as the anchor player or as a trajectory feature.
- Estimate whether the anchor is the first return by looking for incoming shuttle motion towards the anchor player.
- Require simple path structure so a shrinking distance caused by a wild hallucination does not pass by itself.
- Try both a plain direction description and a structured path comparison.
- Treat curve-fit measurements as diagnostics unless they add clear value.
- When motion identifies a return, infer the other player as server.
- Run a second experiment that prepends one missing contact and calls the existing alternating fit once on the augmented sequence.
- Use no fabricated serve frame. Temporal localisation remains a later experiment.
- Evaluate all three videos in full. Show all rallies and the frozen failure subset: 99 covered rallies with a wrong released server label plus 22 with no released label.
- Use all three videos for this EDA. Do not add a train/test split.
- Prefer existing code, NumPy, pandas, scikit-learn and Matplotlib.
- Make dedicated scripts only. Do not change `src/**`.
- Track useful scripts, tests and documents. Ignore external inputs, generated results, plots, case images and delegated-agent records.
- Follow `.github/AGENTS.md`: `.npy.xz`, `.json.gz` and `.csv.gz` for generated data.
- Use plain Australian English. Put the core account of each document within its first 800 words.
- Work only on `investigation/serve-start-trajectory`.

## Choices used in the run

- Use a 30 base-30-frame lookback, matching the original question, rather than the production serve rule's 25-frame lookback.
- Use unambiguous GT contact-1 versus contact-2 anchors to choose the exploratory first-return threshold. Then report server attribution across every rally without relabelling ambiguous or unmatched anchors as contact 1 or 2.
- Show a second threshold curve for final server macro-F1, but do not use it as the main threshold-selection rule.
- Record earlier rejected raw candidates as diagnostics. Do not veto a case merely because a rejected impulse exists.
- Abstain when direct anchor attribution is `None`.
- In Experiment 2, show both the parity-only and player-labelled prepend. This separates the missing-contact effect from the new player vote.
- When a direct anchor half exists but no qualifying path exists, show both a forced anchor-player attribution and an evidence-only abstention.

## User approval on 10 August 2026

- The 30-base-30-frame maximum is only a search limit. The question is whether any usable incoming path appears before the anchor.
- Choose the displayed threshold by first-return F1 on unique contact-1 and contact-2 matches. Explain this in plain language and show every count.
- Include earlier rejected impulse candidates in the analysis rather than using them as an automatic veto.
- Show both the parity-only and player-labelled prepends, with a clear explanation of the difference.
- Show both the forced anchor-player result and the evidence-only abstaining result when no usable path exists.
- The four proposed commit messages in `plan.md` are approved for this local feature branch.
