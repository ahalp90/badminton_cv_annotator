# Final deployment view: contacts, serves and automatic use

The ranking model selects whole-rally clips reliably. Their contact annotations still need review.

## The two reads used throughout this report

ShuttleSet22 contains **3,965 rallies** in the source CSVs.

The existing cleaning drops **543** from strict scoring: 542 contain at least one contact marked `flaw`, and one has timestamps out of order.

The tables compare two sets of ground-truth (GT) labels:

- **Trusted GT only:** score against the remaining 3,422 rallies.
- **All GT included:** restore the 543 rallies and their source labels. Unknown selections receive no credit.

Both use the same saved predictions, scored at **±10 frames on a 30 fps clock**.

## Contact performance

Across the 47 videos:

| Task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Timing only | Trusted GT | 81.0% | 88.2% | 84.5% |
| Timing only | All GT | 90.1% | 86.9% | 88.4% |
| Timing + correct player | Trusted GT | 78.5% | 85.5% | 81.8% |
| Timing + correct player | All GT | 87.2% | 84.0% | 85.6% |

Serves are harder to find than later contacts. The next section separates serve detection from getting the rally start right.

![All-contact precision, recall and F1.](figures/contact_prf.svg)

Contact breakdown: [contact_performance.md](contact_performance.md).

## Serve performance

There are two useful serve questions.

### Is the serve found anywhere in the contact stream?

This is a recall measure because the full-stream detector outputs contacts, not a dedicated “serve” prediction.

| Measure | Trusted-GT recall | All GT included |
|---|---:|---:|
| Serve timing | **2,781 / 3,422 = 81.3%** | **2,855 / 3,965 = 72.0%** |
| Serve timing + correct server | **2,647 / 3,422 = 77.4%** | **2,667 / 3,965 = 67.3%** |

### Does the proposed rally start at the serve?

Every nonempty proposal makes one explicit start prediction, so this task has clean precision, recall and F1:

| Task | Labels | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Start is serve | Trusted GT | 70.4% | 76.7% | 73.4% |
| Start is serve | All GT | 72.1% | 67.7% | 69.8% |
| Start + correct server | Trusted GT | 68.1% | 74.1% | 71.0% |
| Start + correct server | All GT | 68.5% | 64.4% | 66.4% |

The sequence-based player assignment still helps substantially. Among the 2,781 serves matched in time:

| Player answer | Correct | Wrong | Missing |
|---|---:|---:|---:|
| Raw wrist/net guess | 2,222 | 250 | 309 |
| **Final sequence-based answer** | **2,647** | **128** | **6** |

This table checks player assignment only for serves already found within the timing window.

## Automatic use: fully correct rallies

The ranking model selects **784** proposed rallies. Of these, **740 can be judged against trusted GT** and **44 cannot**.

| Measure | Trusted GT only | All GT included |
|---|---:|---:|
| **Fully correct rally precision** | **616 / 740 = 83.2%** | **615 / 784 = 78.4%** |
| **Fully correct rally recall** | **616 / 3,422 = 18.0%** | **615 / 3,965 = 15.5%** |
| **Fully correct rally F1** | **29.6%** | **25.9%** |

Restoring the original labels gives **615 correct, 140 wrong and 29 still unknown**. Unknown cases receive no credit in the all-GT column.

The selector also leaves **1,147 fully correct trusted-GT rallies unselected**.

For fully correct rally selections, the selected set still needs review. Keep automatic approval off.

## Automatic use: contains one whole rally

The exact metric hides an important product result.

Of the **124 selected proposals that fail strict scoring**, **112 still contain exactly one whole labelled rally**. Their problem is inside the contact sequence.

Only **12** have a fundamental rally-level problem: they cut off a rally or overlap more than one labelled rally.

That gives:

| Measure | Trusted GT only | All GT included |
|---|---:|---:|
| **Contains one whole rally precision*** | **728 / 740 = 98.4%** | **739 / 784 = 94.3%** |
| **Contains one whole rally recall*** | **728 / 3,422 = 21.3%** | **739 / 3,965 = 18.6%** |
| **Contains one whole rally F1*** | **35.0%** | **31.1%** |

*Some contact details inside the rally may still be incorrect.*

![Fully correct rally versus whole-rally selection, using the same two reads.](figures/automatic_use.svg)

### What is wrong inside the 124 fully-correct-rally failures?

These categories overlap:

| Problem | Selected proposals |
|---|---:|
| Extra predicted contact(s) | **92** |
| Misses the serve | **43** |
| Misses a later contact | **39** |
| Wrong or missing player assignment | **10** |
| Cuts off a rally or overlaps more than one rally | **12** |

![Why the selected-but-imperfect rallies fail strict scoring. Trusted GT only.](figures/selected_errors.svg)

The key result is **112 / 124 = 90.3%**: most fully-correct-rally failures are still the right whole rally with local contact errors.

## The 44 clips with untrusted GT

All 44 received a sampled visual review. Four mix replay footage with live play, and one appears to be warm-up footage. The other 39 show live play, though many openings are obscured by camera changes.

Two frames per second cannot verify exact contact timing, so this adds no perfect-rally credit. [Review details and clip notes](promising_leads.md#4-what-the-44-untrusted-gt-selections-contain).

## Deleting extra contacts: not worth another model

A separate deletion score was tested on the development videos.

It raises perfect rallies **1,209 → 1,217**, but that comes from **22 repairs and 14 losses**.

That trade is too weak. The model was not carried into the 47-video run.

## Deployment recommendation

For **fully correct rally selection**: keep automatic approval off.

For **review ordering**: keep the ranking score.

For **selecting clips that contain one whole rally***: the ranking model is already very strong, at **98.4% precision with trusted GT and 94.3% verified with all GT included**.

Compact numbers and reproduction command: [serve_tables.md](serve_tables.md).
