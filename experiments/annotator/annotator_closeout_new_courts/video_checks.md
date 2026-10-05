# Label and video checks

**The checked labels support the main diagnosis of model errors.** The largest retained label problem is in video 12: four long rallies whose official timestamps sit about 15 frames off the visible hits. They hold 101 misses, and setting them aside moves fully correct rallies from 52.4% to 52.5%. Elsewhere in the largest failure groups, every checked case backs the official label. Video 15 stays excluded.

**Contents**

[Video 15 stays excluded](#video-15-stays-excluded)  
[Video 53 holds up](#video-53-holds-up)  
[Video 12: four rallies with shifted timestamps](#video-12-four-rallies-with-shifted-timestamps)  
[Random missed-contact sample](#random-missed-contact-sample)  
[Are the large failure groups label errors?](#are-the-large-failure-groups-label-errors)  
[Where the labels come from](#where-the-labels-come-from)  
[Original ShuttleSet](#original-shuttleset)  
[Evidence](#evidence)

## Video 15 stays excluded

Video 15's labels refer to the wrong rallies across all three games, so [#147](https://github.com/ahalp90/badminton_cv_annotator/issues/147) drops it. The [earlier label checks](../../../scratch/annotator_wrapup_evaluation/video_checks.md#video-15-exclude-it) give the evidence.

## Video 53 holds up

Video 53's labels passed the earlier game, score and hit checks. The court stage caused its earlier failure. Compared with the historical model, fully correct rallies rise from 7 to 32 and matched hits from 195 to 880. [Court checks](court_checks.md#video-53-the-scene-is-accepted) give the detail.

## Video 12: four rallies with shifted timestamps

Four long rallies in game 1 have official timestamps that sit off the visible hits as a block. Shifting each rally's labels by one fixed amount recovers almost every hit:

| Rally | Labelled hits | Matched | Labels sit… | Matched after the shift |
|---|---:|---:|---|---:|
| Game 1, rally 3 | 38 | 7 | 7–19 frames early | 38 |
| Game 1, rally 8 | 33 | 6 | 15–18 frames late | 31 |
| Game 1, rally 17 | 34 | 5 | 14–19 frames late | 34 |
| Game 1, rally 19 | 33 | 19 | 16–27 frames late | 33 |

The ranges are the shifts that recover the most hits.

The footage agrees. One hit was checked in each rally. In all four, the labelled player does not hit within ten frames of the label; the nearest visible hits sit 13–16 frames away. In three, the model's prediction is within three frames of the visible hit. The rallies themselves are right: during rally 17 the scoreboard shows An 10–6 Pusarla, and the official row records 11–6 after it.

Together these rallies hold **101 of video 12's 128 misses**. The rest of video 12 scores normally: 29 of 57 rallies fully correct and 27 misses in 643 labels. The [earlier report's](../../../scratch/annotator_wrapup_evaluation/video_checks.md#other-weak-videos) late label at frame 18,232 is in rally 8.

No other rally shows the same pattern. A search over every rally found no other case where shifting the labels by eight or more frames recovers five or more hits.

Setting the four rallies aside barely moves the totals:

| Result | All 46 videos | Without the four rallies |
|---|---:|---:|
| Fully correct rallies | 1,744 / 3,327 (52.4%) | 1,744 / 3,323 (52.5%) |
| Contact timing matches | 34,200 / 37,184 (92.0%) | 34,163 / 37,046 (92.2%) |
| Timing + correct player | 33,156 (89.2%) | 33,153 (89.5%) |
| Missed contacts | 2,984 | 2,883 |

The found timing errors are too small to justify a separate headline evaluation with those rallies removed.

## Random missed-contact sample

The hit and player agree in 21 of 24 randomly sampled misses. The other three have wrong timing labels. Each case was judged from stills around the labelled frame.

![Direct checks of 24 randomly sampled missed contacts, split by court decision.](figures/contact_sample_results.png)

| Direct judgement | Court accepted | No court found | Court found; player check failed | Total |
|---|---:|---:|---:|---:|
| Hit and player agree | 12 | 6 | 3 | **21** |
| Timing label wrong | 3 | 0 | 0 | **3** |
| Clear footage disagreement | 0 | 0 | 0 | **0** |
| Unclear | 0 | 0 | 0 | **0** |

Two of the three timing errors are video 12's shifted rallies 17 and 19. The third is a video 35 serve labelled about 14 frames early, a borderline case.

Outside video 12, 1 of 22 sampled misses has a wrong label. With a sample this small, the true share could still be as high as about one in five (95% upper confidence bound).

No sampled case shows a wrong rally, and all 24 labelled frames show full-court live play with both players in view. The match rates give a stronger collection-wide check: every game of every included video matches at least 72.9% of its labelled hits, while video 15 matched 16% under the historical model.

All nine misses in rejected scenes show the labelled hit. Those misses belong to the court stage, not the labels.

## Are the large failure groups label errors?

Three large failure groups could have been explained by label errors at scale. The checks used three runs of a large vision-language model, which can interpret images and text. Each run saw stills around one frame without being told whether the frame came from a label or a prediction.

![Bulk failures checked against the footage: 26 of 28 cases back the label.](figures/label_check_results.png)

| Failure group | Rallies | A label error would look like… | Cases checked | Label right | Label wrong |
|---|---:|---|---:|---:|---:|
| Extra hit after the last label | 262 | a real final touch the labels skip | 16 | **16** | 0 |
| Swapped sides, outside video 12 | 10 | the official hitter on the wrong player | 10 | **10** | 0 |
| Video 12 shifted rallies | 4 | timestamps off the visible hits | 2 | 0 | **2** |

### Extra hit after the last label

The sample comes from three successively narrower groups:

- **363 failed rallies** have every labelled contact matched with the right player.
- **262 of those rallies** have an extra contact 10–60 frames after the last label, assigned to the other player.
- **233 of those 262** fail on that contact alone: 14.7% of all 1,583 failed rallies. The other 29 have further extras or share a clip with a neighbouring rally.

The footage shows no such hit in any of the 16 rallies sampled from the 262. All 16 are among the 233:

- in 11, the shuttle lands untouched or drops off the net;
- in 5, a player catches the shuttle, picks it up or bounces it on the racket after the rally has ended.

The official rows agree. Of the 233, 230 record an ending that rules out another touch: shot out, into the net, short of the net, a winner landing or a misjudged landing. One records a fault and two record no ending.

The sampled extra hits are therefore model errors. With 0 label errors in 16 cases, the share of these rallies hiding a real final hit is below about one in five (95% upper confidence bound).

Shots into the net and misjudged landings fail this way more often. 10.0% of rallies ending in the net and 13.5% ending in a misjudged landing fail on the extra hit alone, against 5.4% of rallies ending with a shot out. The false hit tends to come when the receiver reacts to a dead or dropping shuttle.

The footage checks establish that the added events in these 16 cases are false contacts.

### Swapped sides

In 13 rallies, the model gives at least 80% of the matched hits to the other player. Three are video 12's shifted rallies. One hit was checked in each of the other ten.

In all ten, the labelled player hits. Five of these rallies are in video 17's 40-minute scene. In seven of the ten, the model gives a far-player hit to the near player. These are model side errors.

## Where the labels come from

The cleaned test labels copy the official ShuttleSet22 rows. All 37,184 labels are official rows with the same frame and rally. Cleaning drops whole rallies that contain a flawed, invalid or out-of-order row: 524 of 3,851. It never moves or drops a hit inside a kept rally.

The labelled side comes from the official foot positions: the hitter is `Top` when their annotated feet are higher in the image than the opponent's. Those positions are image pixels on a 1280×720 grid. Scaled by 1.5, they land on the players' feet in the 1920×1080 source frames, and the official court corners land on the court corners.

A timing or side error in the cleaned labels therefore comes from the official files, not from cleaning.

## Original ShuttleSet

This round adds no footage checks for original ShuttleSet. The 32 held-out development videos score lower on serves than ShuttleSet22, and [#77](https://github.com/ahalp90/badminton_cv_annotator/issues/77) already records first-stroke labels that do not mark the actual serve contact. [Original ShuttleSet numbers](evaluation_tables.md#original-shuttleset) gives the counts.

## Evidence

- `results/miss_sample.csv.gz` and `results/miss_judgements.csv.gz` — the random missed-contact sample
- `results/label_check_cases.csv.gz` and `results/label_check_judgements.csv.gz` — the failure-group cases and verdicts
- `results/label_trace.csv.gz` — the official rows behind every checked case
- `results/tail_rallies.csv.gz` — the 262 rallies with an extra hit after the last label, with extra counts and official endings
- [Footage checks](evaluation_reproduction.md#footage-checks) — how the cases were drawn, rendered and judged
- `raw/miss_frames/` and `raw/label_frames/` — local stills, Git-ignored
