# Current decisions

These choices define the completed investigation. Change them only in a clearly labelled follow-up experiment.

## Rally groups

- **239 one-to-one rallies:** main set for anchor identity, trajectory classification and server attribution.
- **249 covered rallies:** sensitivity check for the current `COVERED` definition, including five predicted spans that each cover two GT rallies.
- **292 GT rallies:** end-to-end view that includes segmentation failures.

Merged predicted spans are not double-scored in analyses that assume one predicted contact sequence per GT rally.

## Contact alignment

- ±10 base-30fps frames is the main usable tolerance.
- ±5 is the strict sensitivity check.
- ±30 is a broad sanity check.
- At every tolerance, keep the nearest GT stroke ordinal, signed offset and absolute offset.
- Flag multiple GT strokes in the window separately. Do not replace the nearest-stroke label with an ambiguity bucket.
- For an earliest contact unmatched at ±10, inspect later accepted contacts for the serve, then the first return, and record the rank of the first GT match.

Use “GT-incompatible candidate under the ±30 sanity check” unless a case has been visually verified. Do not call an unverified candidate a false contact.

## Motion rule

The new rule fits a robust trend to shuttle-to-player distance. It calls the path incoming only when the fitted distance decreases by at least **0.05 apparent player body heights** over the observed path.

The 0.05-BH value is an engineering judgement chosen before corrected scoring. It is not a calibrated physical constant. Do not sweep or retune it in this investigation.

Report fitted decrease, residual scatter and trend-to-jitter as continuous diagnostics. They do not make the call.

Keep the historical 0.25-BH closure plus 55%-towards rule unchanged as provenance. Its thresholds were not independently established physical cut-offs: the 55% value came from the older ±5/249 exploration, and the origin of 0.25 BH was not established.

Apply the same fixed rule unchanged to recurrence-only and recurrence-plus-producer-mask paths.

## Server comparison

Separate these methods:

1. released alternating fit;
2. earliest-contact player;
3. direct 0.05-BH motion correction with earliest-contact fallback;
4. prepend/refit 0.05-BH correction with the same fallback;
5. old-fit-fallback prepend/refit, retained only as an earlier exploratory measurement.

Do not present 163 to 127 as the effect of refitting. The fair direct versus prepend/refit comparison is 163 versus 159 on the same 239 rallies.

## Scope

Keep `src/**`, frozen ground truth, source fixtures, segmentation behaviour and unrelated experiments unchanged. This investigation does not add models or annotations.
