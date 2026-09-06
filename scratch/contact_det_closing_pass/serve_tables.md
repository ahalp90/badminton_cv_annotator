# Contact, serve and automatic-use numbers at a glance

## The two reads

- **Trusted GT only:** 3,422 ShuttleSet22 rallies.
- **All GT included:** all 3,965 rallies in the source CSVs.

The 543-rally difference is 542 rallies with at least one contact marked `flaw`, plus one rally with timestamps out of order.

## Individual contacts

| Contact task | Precision | Recall | F1 |
|---|---:|---:|---:|
| Timing only | **81.0%** | **88.2%** | **84.5%** |
| Timing + correct player | **78.5%** | **85.5%** | **81.8%** |

Recall by labelled contact type:

| Contact type | Timing recall | Timing + correct-player recall |
|---|---:|---:|
| Non-serve | **88.9%** | **86.3%** |
| Serve | **81.3%** | **77.4%** |

Full-stream non-serve precision/F1 is not reported because the detector does not assign every prediction a serve/non-serve class.

## Final detector

| Task | Precision | Recall | F1 |
|---|---:|---:|---:|
| Proposed start is the serve | **70.4%** | **76.7%** | **73.4%** |
| Proposed start is serve + correct server | **68.1%** | **74.1%** | **71.0%** |

Additional recall-only measures:

| Measure | Trusted GT only | All GT included |
|---|---:|---:|
| Serve found anywhere | **81.3%** | **72.0%** |
| Serve found anywhere + correct server | **77.4%** | **67.3%** |
| Perfect whole rally | **51.5%** | **44.5%** |

The [reproducible table](metric_summary.md) includes all-GT contact and start scores, with the tighter ±5 check.

## Player assignment after the serve is found

This diagnostic uses the 2,781 trusted-GT serves found in time.

| Player answer | Correct | Wrong | Missing |
|---|---:|---:|---:|
| Raw wrist/net guess | 2,222 | 250 | 309 |
| **Final sequence-based answer** | **2,647** | **128** | **6** |

## Automatic use: fully correct rallies

| Measure | Trusted GT only | All GT included |
|---|---:|---:|
| **Fully correct rally precision** | **83.2%** | **78.4%** |
| **Fully correct rally recall** | **18.0%** | **15.5%** |
| **Fully correct rally F1** | **29.6%** | **25.9%** |

## Automatic use: contains one whole rally*

| Measure | Trusted GT only | All GT included |
|---|---:|---:|
| **Contains one whole rally precision*** | **98.4%** | **94.3%** |
| **Contains one whole rally recall*** | **21.3%** | **18.6%** |
| **Contains one whole rally F1*** | **35.0%** | **31.1%** |

The 728 trusted-GT successes are:

- 616 perfect;
- 112 correct whole rallies with contact-level mistakes.

Only 12 of the 124 fully-correct-rally failures have a fundamental rally-level problem.

The all-GT recount uses the original labels: 615 selections are fully correct, and 739 contain a whole labelled rally. Cases without enough labels receive no credit.

## Why the 124 fully-correct-rally failures fail

These counts overlap:

- 92 have extra contact(s);
- 43 miss the serve;
- 39 miss a later contact;
- 10 have a player-assignment error;
- 12 cut off a rally or overlap more than one rally.

![Automatic-use results.](figures/automatic_use.svg)

Machine-readable per-video serve table: `results/serve_followups/serve_per_video.csv.gz`.

*Some contact details inside the rally may still be incorrect.*
