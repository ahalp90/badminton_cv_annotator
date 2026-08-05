# Trial feature list

This draft records the trial candidate outputs for downstream player-level analysis.

The list summarises [issue 13](https://github.com/ahalp90/badminton_cv_annotator/issues/13),
its [pipeline feasibility review](https://github.com/ahalp90/badminton_cv_annotator/issues/13#issuecomment-5099513256),
and its [feature-structure guidance](https://github.com/ahalp90/badminton_cv_annotator/issues/13#issuecomment-5161351662).
The issue remains the source for research references and detailed rationale.

## Structure

- **Extractor primitive:** A direct pipeline input or output, such as a player
  keypoint, shuttle position, court homography, rally span, or contact frame.
- **First-order feature:** A simple calculation from primitives, such as a
  player's distance from half-court centre at an opponent contact.
- **Higher-order feature:** An aggregate or combination of simpler features,
  such as shot frequency or movement efficiency.

All rally-derived values should be limited to high-confidence, complete
rallies unless a feature defines another validity rule. Stored values should
include source provenance, frame or time coordinates, confidence, and validity
flags where available.

## Trial feature shortlist

| Feature | Layer | Suggested stored value | Provisional value | Readiness and main limit |
| --- | --- | --- | --- | --- |
| Rally duration | Extractor primitive | One duration per rally | Medium | Current candidate. Depends on a high-confidence, complete rally span. |
| Out-of-position posture states | First-order | Per player and rally, proportion of time in Dive, Off-balance, and Stretch states | High | Current pipeline candidate. The state definitions and any contact-frame dependency remain open. |
| Away-from-centre recovery position | First-order | Distance from half-court centre at each known opponent contact | High | Noisy candidate. Needs usable homography and contact detection. Any per-rally aggregate or player ratio remains open. |
| Rest time, work density, and effective playing time | Higher-order | Rally and inter-rally timing measures, with match aggregates where valid | Low, uncertain | Noisy candidate. Broadcast cutaways complicate gaps, and work density still needs a precise definition. |
| Smash shuttle speed | First-order | One speed per detected smash event | Medium, uncertain | Noisy candidate. Needs smash classification, contact frame, shuttle track, homography, and frame rate. Missing smashes are expected. |

Event-level values should be retained when possible. Match-level or player-level
summaries can then be derived without hiding missed events or uneven rally
coverage.

## Deferred until upstream reliability improves

| Feature | Layer | Reason for deferral |
| --- | --- | --- |
| Match duration | Higher-order | Missing rallies and non-match footage can distort the value. |
| Shots per rally | First-order | Contact detection currently misses or adds events. |
| Shot frequency within rally | Higher-order | It combines contact-detection and rally-span errors. |
| Aggression markers | Higher-order | It needs reliable whole-rally contact detection and shot classification. |
| Rally-length distribution by outcome and final landing zone | Higher-order | It needs reliable rally outcome, final contact, and landing estimates. |

## Needs definition before recommendation

| Feature | Layer | Definition work needed |
| --- | --- | --- |
| Stroke duration | First-order | Define motion onset and confirm that it can be validated from pose and contact frames. |
| Court coverage near the shuttle | Higher-order | Define the relative measure, event anchor, and recording frequency. |
| Split-step stance geometry | First-order | Define the stance measure, event detector, and recording frequency. |

## Extractor primitives to preserve

- Rally start and end frames, duration, completeness, and confidence.
- Contact frames, player assignment, and confidence or suppression reason.
- Player keypoints and court-projected player positions.
- Shuttle positions, visibility, and track provenance.
- Court geometry and homography validity.
- Accepted stroke classes and their confidence where classification runs.
- Video frame rate and source timestamps.

## Outside the current trial

Net-game share and clear share need reliable shot classification across
complete rallies. They are outside the current time and infrastructure
constraints.

Backhand proportion, forced-to-unforced error ratio, shot-outcome success by
type, footwork-to-shot coupling, hit height, and shot-selection deception are
excluded. They need new extraction capability, expert judgement, dependable
upstream outputs, or evidence beyond the current project scope.

Source provenance, player and match identifiers, professional or amateur
status, timestamps, names, contact frames, and winner fields are schema and
bookkeeping requirements. They are not performance features.

## Decisions still needed

1. Define the three out-of-position states in measurable pose terms.
2. Choose whether recovery position is stored only per event or also as a
   per-rally mean, median, spread, or player ratio.
3. Define work density and its rally or match denominator.
4. Set validation thresholds before any candidate is described as reliable.
