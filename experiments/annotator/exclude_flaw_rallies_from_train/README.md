# Excluding unreliable rallies from whole-rally training

Excluding unreliable rally labels improved validation but reduced performance
on the held-out test set. The experiment is archived; the main pipeline,
models, scoring and v7 report retain their pre-experiment state.

## What changed

Starting from `refactor/annotator-scoring-fix` at `2c888619`, rallies were flagged
from the original ShuttleSet set CSVs if they contained any flaw flag, had fewer
rows in `shots_master.csv` than in the original, or had consecutive contacts
assigned to the same player. The combined rule excluded 866 of 2,691 training
rallies (32.2%) and 321 of 668 validation rallies (48.1%).

All video evidence and contact labels remained available. Contact trees and
contact scoring were unchanged. Whole-rally targets became −1, which the rally
selector and review fitters skip. Merging multiple rallies still produced a
negative target before the exclusion check. Whole-rally validation and confidence
selection excluded the flagged spans. ShuttleSet22 test scoring retained the
v7 method unchanged.

## Results

Five seeds, ±10-frame contact tolerance. Both validation rows below use the same
347 judgeable rallies and all 5,696 contact labels. Each test comparison uses
the same 3,327 scored rallies. Validation and test use different exclusion rules;
their absolute percentages are not directly comparable.

| Fully correct rallies | Original fit | Exclusion refit | Change |
| --- | ---: | ---: | ---: |
| Validation, same 347 rallies | 64.03% | 65.48% | +1.44 points |
| Test, supplied rally bounds | 60.34% | 55.73% | −4.61 points |
| Test, discovers its own rallies | 54.96% | 51.93% | −3.02 points |

The validation gains were 1, 7, 9, 5 and 3 rallies across the five seeds.
Losing roughly a third of whole-rally supervision is a plausible explanation
for the test regression, but this run does not separate data volume from label
quality or the suitability of the unchanged tree settings. The finding applies
to the combined exclusion rule, rather than flaw flags alone.

MonoTrack's 12 configurations × five seeds were rescored on the 347 validation
rallies. The winner remained zeroed shuttle input, L2=0, interpolation off
(68.18% fully correct); its training and test predictions were reused.

The refit's confidence survey searched all 1,939 attainable validation cutoffs.
The F1-selected cutoff was 0.478665: 2,246 test detections retained per seed,
80.0% with correct boundaries and 65.9% fully correct. Keeping half of validation
detections selected 0.529525: 2,063 retained on test, 81.0% with correct boundaries
and 67.8% fully correct. No cutoff reached 99% mean validation boundary accuracy
with a non-empty selection in every seed. Boundary figures use the 2-second-before /
3-second-after allowance.

## Files

- [Implementation patch](implementation.patch): exclusion logic, fitting/scoring
  changes, regression tests and the three cache-reuse/reporting runners. Apply
  only in a separate checkout of `2c888619`.
- [Exclusion builder](build_rally_exclusions.py) and its
  [tests](test_build_rally_exclusions.py): runnable on the restored main code.
  The builder writes only this experiment's result directory by default.
- [Results](results/): original/refitted per-seed test summaries, the common-rule
  validation comparison, exclusion counts/list, MonoTrack selection and both full
  confidence surveys. These are independent of the experiment branch.
- [Reproduction](reproduce.md): reuse the existing prepared evidence and contact
  fits when rerunning the experiment. Large model bundles and raw predictions
  are not duplicated here.
