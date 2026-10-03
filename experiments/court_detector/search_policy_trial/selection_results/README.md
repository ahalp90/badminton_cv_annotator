# Choosing courts by score instead of player checks

Choosing the highest-scoring court improved some individual detections but
made no difference to the final main-camera courts after sharing. It also
produced false courts in frames where there was no court to find.

This comparison used eight videos: ShuttleSet 11, 21, 30 and 36, and ShuttleSet22
27, 43, 44 and 51. Both methods used the same search results. The alternative
selected by court score and used player positions only to break exact ties;
it also removed the player check after the final adjustment of the court.

## What changed

Errors below compare corners with the official main-camera labels at 1280 × 720.
Both methods are measured on the same 726 scenes from the main camera views.

| Result | Original player checks | Score-first |
| --- | ---: | ---: |
| Individual scene fits with every corner within 20 px, before sharing | 706 / 726 | 721 / 726 |
| Final scene fits with every corner within 10 px, after sharing | 726 / 726 | 726 / 726 |
| Scenes with any detected court, across all 4,615 scenes | 1,263 | 1,317 |

Before sharing, score-first repaired 16 fits and broke one. That failure was
ShuttleSet 11 scene 154: the worst corner difference rose from 9.4 to 1,418.6
pixels. Its oversized court scored 0.807, above the accurate court's 0.687.
Sharing then restored an accurate court. Looking only at the final results
would hide that mistake.

There were seven exact score ties among 1,660 recorded choices. Player support
did not change any of them, so this trial found no benefit from that tie-breaker.

Score-first added detections in 151 scenes and removed them in 97. Most changes
were outside labelled rallies: 142 additions and 93 removals. Two newly detected
courts were also shared with other scenes, spreading the effect of those choices.

## What the review images show

None of the 16 images with detections contained a real court.
The four other review images had no detected court; leaving them that way was
judged reasonable.

The [20 images](review_frames) contain:

- 01–08: a court added by score-first, one example per video.
- 09–16: a court retained by the original method but removed by score-first,
  one example per video.
- 17–20: frames within labelled rallies where neither method detected a court.

All 16 false-court examples were outside labelled rallies. They were selected
to inspect changes, so their count cannot estimate how often either method
makes this mistake across a whole broadcast.

## Results files

[Per-video results](per_video.csv.gz) and [paired scenes](paired_scenes.csv.gz)
contain the numerical comparisons. The [trial report](../README.md) covers the broader search without player
rejection. The [release evaluation](../../released_dataset_evaluation/README.md)
covers the later sharing repair across all 86 videos. The
[reproduction guide](../../reproduce.md#compare-the-three-search-methods)
rebuilds the trial tables.
