# Choosing one court for each rally

A rally can cross a camera cut. This check asked whether taking its court from
the camera view shown for the longest time changed the original evaluation.
It barely did: only two selected scenes changed, and the number of rallies
matching the official court stayed at 5,228 of 6,833.

These are the original detections, before the sharing repair. The
[current results](../../report.md) include that repair.

## How the court was chosen

A rally runs from its first labelled contact through its last. Scenes are
grouped by camera view, and the group covering most of the rally supplies the
court. Within that group, the chosen detection is the one closest to the other
courts, giving longer overlaps more weight. A group with no court counts as a
miss. The official labels do not influence this choice.

The chosen court is then compared with the official main-camera corners at
1280 × 720. Agreement means the average distance across the four corners is
within 10 pixels. A different camera angle needs its own labels to judge
accuracy.

| Original-run result | ShuttleSet | ShuttleSet22 | Total |
| --- | ---: | ---: | ---: |
| Rallies with usable contact-frame lists | 3,182 | 3,651 | 6,833 |
| Chosen camera group has a court | 3,011 | 3,451 | 6,462 |
| Court agrees within 10 px | 2,631 | 2,597 | 5,228 |

Another 377 rallies were excluded because their contact-frame lists did not
run forward in time. The chosen camera group covers more than half the rally
in 6,829 of the 6,833 included rallies.

The original run also detected courts during 35% of the time outside labelled
rallies. That includes unlabelled play, breaks, replays and close-ups. It says
how often a court was returned, not how often it was correct.

## Files and command

- [per_rally_view.csv.gz](per_rally_view.csv.gz): the selected court and its
  error for each rally.
- [view_populations.csv.gz](view_populations.csv.gz): time covered by different
  camera groups and by detections.
- [summary.json.gz](summary.json.gz): counts, percentages and summaries by video.

This rebuilds the tables from the saved original evaluation:

```bash
PYTHONPATH=.:src python scripts/summarise_court_rally_views.py \
  --input experiments/court_detector/baseline \
  --output /tmp/original-court-rally-views
```
