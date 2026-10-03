# Earlier court-finding approaches

These experiments explain why some tempting fixes were set aside. They used
small development sets, often with settings adjusted after seeing the results.
Their counts are not accuracy estimates for today's detector. The
[current design](../../../docs/court_detector/design.md) and [development story](../report.md)
are the starting points for current work.

## Finding lines is only part of finding the court

The early OpenCV prototype accepted three of 11 labelled amateur frames.
All three courts were wrong. Frozen neural line extractors also failed to
select accurate amateur courts. Better-looking lines did not resolve which
lines belonged to the playing court.

Player positions and net evidence helped some of the three short-clip examples.
They did not establish a reliable acceptance rule. In particular, choosing the
players involved in the rally remains separate from finding people in the image.
The [independent-court experiment](../../annotator/independent_court/README.md)
keeps its implementation and compact results. The
[manual examples](../../../data/amateur_court_corners/README.md) retain useful inputs.

## Better local fits can produce worse final courts

| Attempt | What the comparison showed |
| --- | --- |
| Refit the current best court | Helped four of 19 leaders and harmed none. Applying the same rule across all 357 candidate pairs selected 35 harmful refits. The small success did not generalise even within those saved candidates. |
| Refit each line group with SVD | SVD, a least-squares fitting method, improved some groups. It kept their existing fragment assignments and did not run the full court search. It therefore did not test whether the fragments belonged together. |
| Use each line's midpoint to estimate directions | Recovered useful courts in some views but broke GX frame 0 and Amateur-3. A more precise line representative had similarly mixed results. Neither was a general replacement. |
| Spread the same number of random seeds more widely | Improved nearby geometry on five of seven GX frames, accepted none and worsened both controls. |
| Reserve half the candidate limit for diverse shapes | Improved one of six tested direction pairs and worsened another. This stopped before a full court search; it was not a demonstrated overall improvement. |

The matcher limits how many line assignments reach later scoring. A useful
court can disappear at that limit even when its lines were found correctly.
Changing the limit or its ordering needs a comparison of the final selected
courts across the full candidate search. A win on one direction pair can be
reversed when candidates from other pairs compete.

The recorded Amateur-3 example makes the distinction concrete. A 4.28-pixel
assignment lost to a higher-scoring 8.54-pixel version of the same assignment.
The best distinct assignment had 6.24-pixel error but ranked 11,518th.
The closest one inside the 512-entry limit was 49.65 pixels away.
Deduplication and the candidate limit caused different losses here.

## What is still available

The [independent-court results](../../annotator/independent_court/recorded)
include the compact broadcast, amateur and non-court comparisons, neural line
caches and saved replay bundles. The
[saved-view inputs](../saved_views/README.md)
support current tests and detector comparisons.

Old raw search dumps and recovery archives have been removed. Historical
scripts and detailed reports remain in Git history before this reorganisation;
the retained summaries do not promise a complete rerun of every old experiment.
