# Follow-up 6: dense rally-opening context

## Bottom line

Test whether InternVideo3 can identify the server more reliably when it sees the
whole broadcast transition into a rally opening. Compare all native frames with
every second native frame. Separately test whether plain-language automatic
navigation cues help.

This is a new result. It does not change the completed Follow-up 2 or Follow-up
4 records.

## Why this test exists

Follow-up 2 gave the model 120 consecutive frames near a proposed rally start.
That was about four seconds. The view could begin after preparation or omit the
relationship between an unusual camera view and the normal court view.

The current annotator already proposes rally spans and accepted contacts.
PySceneDetect already records broadcast shot changes. Combining those two
automatic sources gives a deployable way to find openings that may begin in a
non-standard view.

## Persistent join

Build one reusable inference manifest for all 311 retained automatic spans.
Keep its scoring truth in a separate file.

For each automatic span:

1. Inspect its first three accepted contacts, or every contact when fewer than
   three exist.
2. Find shot changes from two seconds before the first contact through the last
   inspected contact.
3. When at least one shot change qualifies, take the complete region from five
   seconds before the earlier evidence through five seconds after the later
   evidence.
4. Retain spans that do not qualify with an explicit reason.

The route uses no ground truth. The later crosswalk retains unmatched labelled
rallies instead of forcing them onto an automatic span.

## Visual range check

Before inference, inspect median and long selected windows from every fixture
as local PNG sequences. Select these examples from the inference manifest only.

Use the range check to answer two questions:

- Does the window contain the non-standard view, the return to court and enough
  early play to relate the views?
- Is native density practical, or is every second native frame the sensible
  upper setting?

## Frozen comparison

Use the routed one-to-one cases that already have independent human visibility
review. This produces 12 cases across all three fixtures. The selection is
truth-filtered, although no labels enter the model prompt or clip.

Give every case one continuous 22-second clip. This duration contains every
routed window. Add real adjacent broadcast footage to make shorter cases the
same length; do not pad with repeated frames.

Run three arms with the same pinned InternVideo3 model:

- **Clean half-native:** every second native frame and no automatic timing
  observations.
- **Cued half-native:** the same video frames plus one plain-language sentence
  giving the shot-change time and a broad range covering the first few possible
  contacts.
- **Cued native:** all native frames with the same cue.

The cue states that possible contacts may be returns or later shots. It does
not include the heuristic server prediction. The prompt also warns that the
close-up player is not necessarily the server.

Ask only for `top`, `bottom` or `unclear`, plus one short evidence sentence.
Score server attribution only.

## Evidence gates

- Use exactly the same 12 case identities in all arms.
- Require the expected complete frame grid: 275 or 330 frames at half-native;
  550 or 660 frames at native density.
- Require zero generation errors and zero parser errors.
- Validate every attempt against the frozen case, prompt and clip hashes before
  scoring.
- Report paired gains and regressions. Do not rely only on the three totals.

Treat a one-case movement as descriptive noise in this 12-case subset. A
practical prompt or density change needs at least two net additional correct
answers without hiding failures behind abstention.

## Interpretation boundary

The clean-versus-cued pair isolates the navigation sentence at half-native
density. The half-versus-native pair isolates frame density with the cue held
constant.

Comparison with Follow-up 2 is descriptive because this trial changes the clip
length, framing, task and prompt together. A strong result would justify a new
wider evaluation and a prospective reassessment of the operational choice. It
would not rewrite the historical Follow-up 2 finding.
