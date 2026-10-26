# Selected discovery model

The default annotator uses the **no-max raw-baseline bundle, seed offset 0**,
refitted with unreliable whole-rally labels excluded. Training and discovery
retain outward context without a fixed time limit.

The 32 training videos retain 1,825 of 2,691 rallies for whole-rally learning.
The eight validation videos retain 347 of 668 rallies for whole-rally scoring.
All 27,571 training contacts and 5,696 validation contacts remain available for
contact fitting and scoring. The contact trees are reused unchanged.

Both the previous and refitted annotators below use the same 347-rally scorer,
five seeds and ±10-frame contact tolerance. Values are five-seed means.

| Fit | Fully correct / 347 | Share | Contact F1 | Correct-player F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous whole-rally labels | 222.2 | 64.03% | 90.136% | 85.336% |
| Unreliable whole-rally labels excluded | 227.2 | 65.48% | 90.211% | 85.411% |

The refit removes unreliable supervision. Its validation gain does not establish
statistical superiority. On the unchanged test protocol, fully correct rallies
fall from 60.34% to 55.73% with supplied bounds and from 54.96% to 51.93%
in discovery. Test results did not select the model or its thresholds.

The [exclusion list](../../../../../data/shuttleset/rally_exclusions.csv.gz)
comes from the original set CSVs: any flaw flag, fewer rows in the master than
in the original rally, or consecutive contacts assigned to the same player.
Excluded rallies retain their video evidence and labels for neighbouring-rally
checks. A merged rally remains a negative target; otherwise an excluded rally
receives -1, which the selector and review fitters skip.

- [Validation comparison](selection.json.gz) contains every seed and the common rule.
- [Model bundles](no_max/models/) contain five raw-baseline and five body-window-36
  models, each with contact, selector and review trees.
- [Fit settings](no_max/fit-settings.json.gz) record selector settings, reused
  contact fits and fixed training/validation videos.
- [40-video contact comparison](no_max/models_40/) combines the existing 40-video
  contact trees with the refitted 32-video selector and review models.
- The [benchmark report](../../../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup_v7.md)
  gives test results and the full confidence threshold surveys.

Discovery assessment uses first and last accepted contacts. Training rejects
merged or incomplete rallies and contact/player errors; extra outward context
alone is allowed. Supplied rally boundaries stay fixed. Confidence reporting
assesses boundaries with 2 seconds before and 3 seconds after labelled contacts,
plus the inward timing allowance. Dataset export separately adds two seconds
before the first accepted contact and three after the last, allowing overlapping clips.

Preprocessing and five-frame-preferred selector targets are unchanged. Maximum
rounds remain 180/400/100 and leaves 31/31/7 for contact/selector/review. Learning
rates are 0.06/0.05/0.05, minimum child weight 5 and L2 regularisation 1. The base
contact seed is 20260824; selector and review use 20260905. Seed offsets are
0, 100, 200, 300 and 400. These vary fitting; they are not independent video samples.
