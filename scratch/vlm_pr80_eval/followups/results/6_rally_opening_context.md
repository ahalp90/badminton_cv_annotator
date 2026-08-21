# Follow-up 6: dense rally-opening context

## Bottom line

Longer dense context did not improve InternVideo3's server attribution on the
12 reviewed routed cases.

The clean half-native arm identified 8 of 12 servers. Adding plain-language
shot-change and contact-range cues reduced this to 7 of 12. Supplying all
native frames instead of every second frame changed none of the 12 cued
answers.

Stop here. Keep the persistent 311-span join for later work, but do not adopt
the timing cue or native-density input. The result does not justify a wider
run or a reassessment of the Follow-up 2 model choice.

## What the model saw

Each case was a continuous 22-second broadcast clip centred on an automatic
rally opening. The clip included the complete routed evidence region: a shot
change near the first few accepted contacts, with five seconds of buffer on
each side. Shorter regions used real adjacent footage rather than repeated
frames.

The experiment used three matched arms:

- **Clean half-native:** every second native frame, with no automatic timing
  observations
- **Cued half-native:** the same frames, plus one sentence giving the
  shot-change time and a range covering the first few possible contacts
- **Cued native:** all native frames, with the same cue

The cue said the possible contacts could be returns or later shots. It did not
include the heuristic server prediction. Every prompt warned that a player
shown in close-up was not necessarily the server.

InternVideo3 received 275 or 330 frames in each half-native case. It received
550 or 660 frames in each native case. The task asked only for `top`, `bottom`
or `unclear`, plus one short evidence sentence.

## Result

| Arm | Correct | Bottom answers | Top answers |
| --- | ---: | ---: | ---: |
| Clean half-native | 8/12 | 10 | 2 |
| Cued half-native | 7/12 | 11 | 1 |
| Cued native | 7/12 | 11 | 1 |

The cue changed one half-native answer. That answer changed from correct to
wrong. It fixed no case.

Native density changed no cued answer. It approximately doubled the
visual-token range from 19,872–23,760 to 39,600–47,520. Total inference time
rose from about 171 seconds for cued half-native to 397 seconds for cued native.

All 36 attempts completed. Every response parsed. The recorded frame grids
contained every expected frame index.

### Results by reviewed serve visibility

| Arm | Visible | Off-frame | Broadcast omitted |
| --- | ---: | ---: | ---: |
| Clean half-native | 4/6 | 2/4 | 2/2 |
| Cued half-native | 4/6 | 1/4 | 2/2 |
| Cued native | 4/6 | 1/4 | 2/2 |

The 12 expected answers were balanced: six top and six bottom. InternVideo3
nevertheless answered bottom in 10 clean cases and 11 cued cases. This small
subset does not establish why that skew occurred.

## Relation to Follow-up 2

InternVideo3 also identified 8 of these same 12 servers in Follow-up 2's
120-frame clean clips. Although both tests scored 8/12 on server identity, the
case-level answers were not stable:

- six cases were correct in both tests
- two were correct only in Follow-up 2
- two were correct only with the new clean half-native clip
- two were wrong in both

This is a descriptive comparison. The new trial changed the clip length,
framing, task and prompt together. The equal 8/12 totals do not prove that
longer context has no value. Four case-level outcomes changed: two previously
correct cases became wrong, while two previously wrong cases became correct.

Keep Follow-up 2's historical 23/32 InternVideo3 server-attribution result
unchanged. This result does not support a new operational preference.

## Persistent join and visual range check

The reusable inference join contains all 311 retained automatic spans:

- 253 have a qualifying shot change near the first three accepted contacts
- 57 have accepted contacts but no qualifying shot change
- one has no accepted contacts

The 253 routed evidence regions range from 10.16 to 21.2 seconds. Their median
is 12.64 seconds. This is far shorter than the initial estimate of up to one
minute. Each model clip expands its routed region to the fixed 22-second input.

The visual audit selected the median and longest routed region from each
fixture before opening the truth file. These six local PNG sequences each
contained a close-up or another view that did not show the full court, followed
by the return to court, preparation and early play. The samples also showed why
close-up identity could not be treated as a server label: the close-up subject
was not consistently the server. The PNGs remain untracked because they are
source-video frames.

The separate truth crosswalk maps 247 labelled rally references to retained
automatic spans and leaves 45 explicit unmatched rows. The scored 12-case
subset requires an automatic span to overlap one labelled rally with a known
server and independent human visibility review. This is a truth-filtered
selection, but labels were absent from the clips and prompts.

## Limits

This is a 12-case diagnostic, not an accuracy estimate for all 253 routed
spans. Fixture representation is uneven: three cases come from `sset_01`, two
from `sset_15` and seven from `sset_21`.

Three of the 12 labelled-to-automatic crosswalks overlap by less than 0.8. The
lowest overlap is 0.524. The evidence retains these fractions so later work can
apply a stricter crosswalk without rebuilding the inference join.

The visual audit covered six selected examples. Those examples fitted within
the 22-second trial input, but the audit did not review all 253 routed regions.
Its visual observations cannot be reproduced from the public package because
the source-video PNGs are deliberately untracked.

## Decision

Retain the persistent join as reusable infrastructure. Its inference rows
contain no labels; the labelled scoring rows are stored in a separate file.

Default to every second native frame in any later trial on these windows.
Native input approximately doubled the visual load and changed no answer.

Do not provide the tested timing sentence. It produced one regression and no
repair.

Do not run the planned wider extension. The experiment spec required at least
two net additional correct answers before widening. Neither change added any.
Keep any future work separate from this completed record.

## Evidence

- [`evidence/6_rally_opening_window_manifest.json.gz`](evidence/6_rally_opening_window_manifest.json.gz)
  contains the truth-free 311-span join and route decisions.
- [`evidence/6_rally_opening_window_truth.json.gz`](evidence/6_rally_opening_window_truth.json.gz)
  contains the separate labelled crosswalk.
- [`evidence/6_rally_opening_trial_manifest.json.gz`](evidence/6_rally_opening_trial_manifest.json.gz)
  contains the 36 frozen model inputs and prompts.
- [`evidence/6_rally_opening_trial_truth.json.gz`](evidence/6_rally_opening_trial_truth.json.gz)
  contains the 12 scoring labels.
- [`evidence/6_rally_opening_score.json.gz`](evidence/6_rally_opening_score.json.gz)
  contains all arm totals, paired comparisons and row-level results.
- [`evidence/6_rally_opening_intern_remote_runs.tar.gz`](evidence/6_rally_opening_intern_remote_runs.tar.gz)
  contains the 36 raw InternVideo3 attempts. Runtime logs are excluded because
  they contain machine-local paths.
- [`../6_rally_opening_context.md`](../6_rally_opening_context.md) records the
  predeclared comparison and widening rule.

The evidence files contain portable labels and hashes. They contain no source
video paths, local usernames or remote-host paths.
