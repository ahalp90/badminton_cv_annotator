# What the ordinary heuristic annotator produces

The ordinary heuristic finds many individual contacts, but very few exact rally
sequences. At ±10 frames and 30 fps, across the same 47 previously examined videos, it
supplies **four fully correct rallies out of 3,422 cleaned labelled rallies**. The final
learned detector supplies 1,763. Both use the same cached court, pose and shuttle
inputs.

The ordinary heuristic uses the hand-written contact rules. The learned detector then
uses trained models to choose contacts and improve the rally sequence. This report asks
how much difference that makes. It reuses saved end-to-end annotation outputs. The
vision models and detector were not changed or retrained.

A contact is one predicted hit. A timing match places it within the allowed number of
frames of a labelled hit. A fully correct rally needs a clip containing the whole
labelled rally, every contact matched once, no extra contact, and the correct player.
The main allowance is ±10 frames at 30 fps. Results below use cleaned labels unless the
section says otherwise. Cleaning has not removed every label error. The cleaned set
contains 38,218 contacts in 3,422 rallies. The all-source check uses 43,159 contacts in
3,965 rallies. All 47 videos have been examined before, so these results do not tell us
how well the system will work on new matches.

The later [follow-up answers](VIDEO_CHECKS.md) add the comparison without videos 15 and
53. That report also contains the newer footage checks.

## Why this is the ordinary output

The job that produced these results first ran the normal annotation code with its
default settings. It saved that output before preparing the learned model's inputs. The
saved files contain the candidate hits, the hits kept after filtering, the clip
boundaries and the assigned players.

The looser impulse and wrist settings come later. They let the learned model search more
possible contact times. The normal heuristic output had already been saved before those
settings were used. Some earlier reports call a learned-model result “original”. Here,
“ordinary heuristic” means the hand-written rules themselves.

This report separates two saved outputs:

- **Heuristic candidates before filtering:** every raw contact candidate. These show what filtering removes; they are
  not the final heuristic annotations.
- **Ordinary heuristic output:** contacts kept by the heuristic's wrist, nearby-event
  suppression and exclusion checks. This is the primary end-to-end comparison.

The player comparison uses the heuristic's own answers. It starts with the saved first
hitter and alternates players across the filtered contacts. The learned model's player
rules and clip-selection rule were not applied to this output. Candidates before
filtering have no saved player answer, so they can only be compared for timing.

The saved records for all 47 videos name the same code version. The relevant annotation
code and default settings have not changed since that run. Loading and scoring the saved
files took about **7.4 minutes**, after a 22-second check on one video. The saved files
supplied what was needed.

## How much does filtering help?

Filtering removes many unmatched candidates, but the remaining contact sequence still
contains too many extras for exact rally annotation.

| Output stage | Contact times listed | Timing matches | Share of labelled hits found | Share of output with a label match |
|---|---:|---:|---:|---:|
| Heuristic candidates before filtering | 98,470 | 30,012/38,218 | 78.5% | 30.5% |
| Ordinary heuristic output | 53,649 | 29,206/38,218 | 76.4% | 54.4% |
| Final learned detector | 41,605 | 33,716/38,218 | 88.2% | 81.0% |

The last two columns ask different questions. The first asks how many labelled hits are
found. The second asks how much of the emitted output has a label match.

![Matched and unmatched event counts before heuristic filtering, after filtering and in the final learned output.](figures/heuristic_contact_comparison.png)

The heuristic's filters remove 44,821 candidates: 12,415 fail the wrist check and 32,406
are removed because they are too close to another candidate. Across these saved files,
those removals account for all the difference between the candidate list and the
filtered output. Filtering also loses 806 timing matches. The number of unmatched events
falls from 68,458 to 24,443.

This is useful filtering, but 24,443 unmatched events remain against the cleaned labels.
A hit without a label match is not necessarily a false detection. Cleaning omitted some
rallies, and video 15 has a confirmed label-to-footage mismatch. Using all source labels
reduces the ordinary output's unmatched count to 21,101.

## Why are there only four fully correct rallies?

Whole-rally correctness is demanding because one extra event is enough to fail an
otherwise good clip. Only five ordinary heuristic clips contain exactly the full
labelled contact sequence at ±10 frames. Four also have the correct players. The fifth
has the wrong player sequence.

| Check | Ordinary heuristic | Final learned detector |
|---|---:|---:|
| Fully correct cleaned rallies, ±10 frames | 4/3,422 | 1,763/3,422 |
| Fully correct cleaned rallies, ±5 frames | 3/3,422 | 1,430/3,422 |
| Fully correct all-source rallies, ±10 frames | 4/3,965 | 1,763/3,965 |
| Fully correct all-source rallies, ±5 frames | 3/3,965 | 1,429/3,965 |

At ±10 frames, all four correct heuristic rallies are also correct in the learned
output. The learned output adds 1,759 fully correct cleaned rallies without losing those
four. At the tighter ±5-frame allowance, two of the three correct heuristic rallies stay
correct and one does not. The overall gain therefore does not preserve every success
under every timing rule.

This comparison includes the final system's contact choices, player handling and clip
boundary adjustments. The gain belongs to that whole set of changes; this comparison
does not show how much each individual part contributes.

The four successful heuristic sequences contain 2, 5, 3 and 10 labelled contacts, in
videos 21, 31, 37 and 50 respectively. The ten-contact example shows that success is
possible beyond very short exchanges. It remains rare in this saved collection.

Across the heuristic's 3,982 proposed clips, the cleaned labels give four correct, 3,035
wrong and 943 unknown. Unknown means there are not enough labels to judge that clip; it
stays separate from both correct and wrong.

A wrong clip can have more than one problem:

| Problem | Wrong clips affected |
|---|---:|
| Extra contacts | 2,899 |
| Missing contacts | 2,358 |
| Wrong player on a matched contact | 2,024 |
| Clip cuts off part of the labelled rally | 454 |
| Clip overlaps several labelled rallies | 70 |

The most common combination is missing contacts, extras and wrong matched players, in
1,516 clips. Another 412 have extras alone, and 370 have missing and extra contacts
without the other listed problems. The complete combinations are in [the result
table](results/heuristic_error_combinations.csv.gz).

Among clips that overlap exactly one cleaned rally, the median clip has two missing
contacts and four extras. Only 70 of these clips have no extra event. This explains why
a reasonably high contact-matching rate does not translate into many exact rallies.

## Does the learned output simply keep every heuristic success?

At the individual-contact level, no. Outside misaligned video 15, the comparison is:

| Timing result for the same cleaned contact label | Contacts |
|---|---:|
| Both outputs match it | 28,351 |
| Only the learned output matches it | 5,200 |
| Only the ordinary heuristic matches it | 660 |
| Neither matches it | 2,973 |

The learned output gains 5,200 matches and loses 660, for a net gain of 4,540. These
counts compare both outputs with the same labels at ±10 frames. The footage was not
checked hit by hit to confirm every apparent gain or loss.

Across all 47 videos, the corresponding counts are 28,486 shared matches, 5,230
learned-only matches, 720 heuristic-only matches and 3,782 misses in both outputs. The
[paired table](results/heuristic_paired_contacts.csv.gz) retains both label sets and
both timing allowances.

## Where do contacts fail within a rally?

![Missed-contact rates for ordinary heuristic and final learned output at rally starts, middles and ends, excluding video 15.](figures/heuristic_contact_position.png)

Serves show the largest gap: the ordinary heuristic misses 1,292/3,327 (38.8%), compared
with 561/3,327 (16.9%) for the learned output. For middle contacts, the missed rates are
20.2% versus 8.2%; for final contacts, 21.8% versus 17.3%.

The figure uses the same cleaned labels outside video 15 for both systems. It counts
misses across the full output, including times rejected by the earlier pipeline stages.
Single-contact rallies count as serves only. The [position
table](results/heuristic_position.csv.gz) also keeps the all-video results and the
tighter ±5-frame check.

Tightening the allowance on all 47 videos reduces ordinary timing matches from 29,206 to
27,466. That loses 1,740 matches. The final learned output loses 744 under the same
change. Ordinary heuristic timing is therefore more sensitive to the allowance as well
as missing more contacts overall.

## How much of the problem is upstream?

Court and tracking failures affect both systems, but they make up different shares of
each system's remaining misses.

Outside video 15, the ordinary heuristic misses 8,173 cleaned labels. The final learned
output misses 3,633. Each column groups its misses by the saved input state:

| Saved input state | Ordinary heuristic misses | Final learned detector misses |
|---|---:|---:|
| Court rejected | 2,541 | 2,374 |
| Court accepted; at least one player pick missing | 69 | 96 |
| Court accepted; both players picked | 5,563 | 1,163 |
| **Total misses** | **8,173** | **3,633** |

Court-rejected frames account for 31.1% of ordinary misses, compared with 65.3% of
learned-output misses. The court inputs did not improve between those columns. The
learned output removes many more errors where inputs are available, leaving the shared
court problem as a larger fraction of its remaining errors.

Both methods use the same saved player inputs. The “player pick missing” row refers
to the exact labelled frame. A predicted hit up to ten frames away can still match
that label. The 69 versus 96 misses therefore come from different chosen contact
times; they do not show that the learned model changed player tracking.

The earlier footage checks help explain two failures before contact scoring: some
rejected scenes show ordinary match play, and a shared court outline loses a clearly
visible far player in two inspected cases. Those checks used the same saved court and
player inputs. They examined selected learned-detector failures and successful
comparisons. They were not a new random sample of heuristic errors.

The [expanded report](REPORT_BIG.md#what-happens-before-contact-scoring) explains the
court checks and their limits. Those checks were not repeated here. They do not tell us
how often the court stage rejects usable play across all videos.

## How reliable are the heuristic's player assignments?

Among the ordinary output's 29,205 timing matches with known target sides, 20,204 have
the right side: 69.2%. The final learned output has the right side on 32,667 of 33,715
such matches: 96.9%. The sets of matched contacts differ, so these percentages describe
each output as a whole. They do not compare player assignment on identical hits.

The ordinary output leaves 1,619 timing matches without a player assignment. Its saved
results contain 386 rally spans without a settled first/last player assignment; 34 spans
have no filtered contacts. Empty spans and unresolved sides remain in the evaluation.

Sides refer to the near and far players in the image. They do not follow an athlete's
identity through court-end changes. These results do not measure how often the system
confuses the athletes' identities.

## What about video 15 and selection?

Video 15 has 195 ordinary heuristic timing matches and 165 learned timing matches
against its 1,034 cleaned labels. The apparent advantage is not trustworthy evidence for
choosing the heuristic there. The label-to-footage mapping is wrong, and even complete
learned timing matches can refer to another rally.

The [expanded alignment checks](REPORT_BIG.md#was-any-of-video-15-actually-fine) found
that all five strong-match windows checked showed a different game or score. Across ten
targeted windows, no labelled section was confirmed aligned. These checks did not cover
all 95 labelled rallies or establish that every emitted clip is unusable.

The rule that keeps 784 learned-output clips depends on the learned model's scores.
The heuristic does not have those scores. Choosing the same clip IDs would impose
the learned output's selection on it, rather than test a heuristic selection rule.
This report therefore scores every ordinary heuristic clip. No replacement rule or
threshold was chosen.

## What this changes about the next step

The ordinary heuristic finds many possible contacts. It also gives us a useful
comparison for the learned detector. But its final rally sequences still need many
contact and player corrections before they can support reliable player-performance
records.

The learned path provides a large improvement in complete rally output on these
previously examined videos. It still needs human review. It fixes many of the
heuristic's errors when usable inputs are available. Court rejection and labels pointing
to the wrong footage therefore account for more of the remaining problem. Those are
useful next checks before training another contact model.

The original detector, cached inputs and selection remain fixed. Scoring preserved
source-frame and label-row identity, including repeated label timestamps. The one-video
check and full recount passed. All label rows matched their corresponding rows in the
learned-output tables. Scripts and compact results are listed in [the data and script
guide](README.md).
