# Rally-start contact model result

## Result

Stop this rally-start follow-up. None of the six fixed choices passed the
training rules, so there is no model to check on the eight validation videos.
The run did not save validation scores or read validation labels.

Every choice found some missing first contacts. Every choice also selected too
many wrong contacts. The required correct-action rate was 80%. The six results
ranged from 18.2% to 51.7%.

| Model | Cut-off | Correct actions | Recovery | New correct sections | Lost correct sections |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.5 | 206 / 1,134 (18.2%) | 206 / 271 (76.0%) | 74 | 118 |
| Logistic regression | 0.7 | 186 / 647 (28.7%) | 186 / 271 (68.6%) | 66 | 40 |
| Logistic regression | 0.9 | 83 / 195 (42.6%) | 83 / 271 (30.6%) | 35 | 4 |
| Shallow HGB | 0.5 | 199 / 748 (26.6%) | 199 / 271 (73.4%) | 68 | 44 |
| Shallow HGB | 0.7 | 177 / 484 (36.6%) | 177 / 271 (65.3%) | 59 | 17 |
| Shallow HGB | 0.9 | 76 / 147 (51.7%) | 76 / 271 (28.0%) | 30 | 0 |

The shallow HGB result at 0.9 was the only choice that added correct sections
without losing an already-correct section. It still made 71 wrong choices out
of 147. Its 51.7% correct-action rate is too far below the written 80% rule to
justify a validation check.

## What this means

The frozen two-candidate list contains useful missing contacts, but these nine
saved inputs do not separate safe additions from wrong additions well enough.
A broad addition rule can improve recall while damaging complete rallies.

Keep the original first contact model without this rally-start addition. Do
not tune another cut-off from these results. Continue to the already planned
all-40-video fit and final test.

## Checks

The two full runs produced equal result bytes and equal held-out score bytes.
The saved detail contains all 5,242 candidates and one held-out score from each
model for every candidate. The action totals were rebuilt from those saved
rows. The saved files contain no machine paths or server details.

The raw files stay outside Git under `raw/rally_start_model/`.
