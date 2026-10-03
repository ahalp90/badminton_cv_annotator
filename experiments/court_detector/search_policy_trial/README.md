# Search trial: do player checks discard useful courts?

This trial tested whether the detector would choose better courts if court
score mattered more than where people were standing. It ran on eight videos,
then shared the detected courts between scenes with matching camera views.

The final main-camera results were effectively unchanged. Removing player
checks found more courts, but the additions included false detections.
A hand review of score-first changes covered eight courts it added and eight
it removed.
None of the 16 review images--which were supposed to have courts--actually had a court.
The [results report](../report.md#did-changing-the-search-help) explains the
counts, review findings and processing time.

## What changed in each method

| Method | Search and selection |
| --- | --- |
| Original player checks | The existing search and player checks. |
| Score-first | The earlier search filters remain. The highest court score wins; player support breaks exact ties, with no final player veto. |
| Broader search | Player-based rejection is also removed during search. Geometry and camera checks remain. |

When court scores tie, the alternatives prefer candidates passing the original
player rule, then candidates with a person in at least half the samples, then
the rest. There is no added player-score bonus. All methods share courts after
searching. The later change allowing courtless scenes to receive a shared court
was not part of this trial.

The videos were ShuttleSet 11, 21, 30 and 36, and ShuttleSet22 27, 43, 44 and 51.
They include known failures and otherwise good videos with difficult scenes.
They were chosen for investigation, rather than as a random accuracy sample.

## Files

- [Selection comparison](selection_results/README.md): what changed when score
  took priority over the final player check.
- `search_results/`: tables comparing all three methods, including processing time.
- [Review images and clips](rally_review/README.md): the courts and missed scenes
  checked by hand.
- `trial_run_video.py`: the experiment runner. It records two choices from each
  search so the alternatives can reuse the same work.
- [`../inputs/player_tiebreak_results/`](../inputs/player_tiebreak_results/): saved
  predictions, choices and timings used by the comparison scripts.

The saved files use these original job names:

| Stage | Folder | Method |
| --- | --- | --- |
| A | `videos/` | Original player checks |
| A | `trial_videos/` | Score-first |
| B | `trial_videos/` | Broader search |
| B | `videos/` | Extra comparison: broader search with the original final player veto |

Each job ran two methods together, so the timings cannot separate their costs.

The [reproduction guide](../reproduce.md#compare-the-three-search-methods)
rebuilds the tables from these saved files.
