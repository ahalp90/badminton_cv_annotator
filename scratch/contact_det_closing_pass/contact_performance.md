# Contact-level performance

The whole-rally score is deliberately strict: one missed contact, one extra contact, or one wrong player makes the entire rally fail.

This page answers the middle question:

**How well does the final detector find individual contacts?**

## All contacts: precision, recall and F1

At ±10 frames, across the 47 videos and **38,218 trusted contact labels**:

| Contact task | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Timing only** | **81.0%** | **88.2%** | **84.5%** |
| **Timing + correct player** | **78.5%** | **85.5%** | **81.8%** |

That comes from:

- **41,605 predicted contacts**;
- **33,716 timing matches**;
- **32,667 matches with both correct timing and correct player**.

With **all source labels** restored, timing P/R/F1 is **90.1% / 86.9% / 88.4%**. Requiring the correct player gives **87.2% / 84.0% / 85.6%**.

At the tighter ±5 check:

| Contact task | Precision | Recall | F1 |
|---|---:|---:|---:|
| Timing only | 79.3% | 86.3% | 82.6% |
| Timing + correct player | 76.9% | 83.7% | 80.2% |

![All-contact precision, recall and F1 at ±10.](figures/contact_prf.svg)

## Serves versus later contacts

The saved full-stream matcher tells us which labelled contacts were serves, so recall splits cleanly:

| Labelled contact | Timing recall | Timing + correct-player recall |
|---|---:|---:|
| **Non-serve contacts** | **30,935 / 34,796 = 88.9%** | **30,020 / 34,796 = 86.3%** |
| **Serves** | **2,781 / 3,422 = 81.3%** | **2,647 / 3,422 = 77.4%** |

![Serve and non-serve recall at ±10.](figures/contact_recovery.svg)

So **serves are the harder contact class**. Non-serve recall is 88.9% for timing and 86.3% when the player must also be correct.

At ±5, non-serve recall is **87.9% timing-only** and **85.5% with the correct player**.

### Why there is no separate non-serve precision here

The full-stream detector outputs **contacts**, not a serve/non-serve class for every prediction. “Serve” becomes an explicit prediction only when a contact is used as the first event of a proposed rally.

So subtracting serves from the label denominator gives a clean **non-serve recall**, but there is no equally natural full-stream “predicted non-serve” denominator.

Rather than invent one, the next section reports precision/recall/F1 for the task that *does* explicitly predict a serve.

## Does the proposed rally actually start at the serve?

Every nonempty proposed rally makes one explicit start prediction. There are **3,725** such starts and **3,422** trusted labelled serves.

At ±10:

| Rally-start task | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Start is the serve** | **70.4%** | **76.7%** | **73.4%** |
| **Start is the serve + correct server** | **68.1%** | **74.1%** | **71.0%** |

At ±5:

| Rally-start task | Precision | Recall | F1 |
|---|---:|---:|---:|
| Start is the serve | 60.0% | 65.3% | 62.5% |
| Start is the serve + correct server | 58.2% | 63.3% | 60.6% |

This is the clean precision-capable serve metric. By contrast, **“serve found anywhere in the full contact stream” is naturally a recall metric**, because the full-stream detector does not mark one arbitrary contact as its serve prediction.

## How contact performance changed during the closing pass

| Detector stage | Timing P / R / F1 | Timing + player P / R / F1 |
|---|---:|---:|
| Whole-sequence model | 81.1 / 87.3 / 84.1% | 75.0 / 80.8 / 77.8% |
| + one later-contact repair | 81.1 / 88.0 / 84.4% | 78.3 / 85.0 / 81.5% |
| **Final detector** | **81.0 / 88.2 / 84.5%** | **78.5 / 85.5 / 81.8%** |

![Contact-level progress during the closing pass.](figures/contact_progression.svg)

The large player-aware gain happens when the later-contact work is added. That stage does more than insert missed contacts: the alternating player sequence also fixes many player labels at timestamps that were already present.

The final rally-boundary fix then adds many **perfect whole rallies** while barely moving contact P/R/F1, because its main job is to make the proposed video interval contain contacts the detector had already found.

## How this fits with the rally-level results

These are different tasks:

- **All-contact detection:** P/R/F1 = **81.0% / 88.2% / 84.5%** for timing.
- **All-contact detection with player attribution:** **78.5% / 85.5% / 81.8%**.
- **Serve recall anywhere in the stream:** **81.3%** timing, **77.4%** with the correct server.
- **Proposed rally starts at the serve:** P/R/F1 = **70.4% / 76.7% / 73.4%**.
- **Perfect-rally recall:** **51.5%**.
- **Contains one whole rally***: **98.4% precision, 21.3% recall, 35.0% F1** on trusted GT.

A detector can have high contact recall while a much smaller share of rallies are perfect end to end. Long rallies multiply the chances of at least one mistake.

## Reproduce the numbers

The [metric summary](metric_summary.md) gives both label populations at ±10 and ±5, plus the command to rebuild them from saved predictions. Full-stream non-serve precision remains undefined because predictions do not carry that class.
