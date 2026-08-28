# Follow-up 2: which VLM is better on reviewed rally starts?

**Result:** InternVideo3 is the better starting model for later VLM comparisons. Intern identified the server correctly in 23 of 32 cases, compared with 14 of 32 for Qwen. Neither model was trustworthy at deciding whether serve contact was visible or at locating the exact contact frame.

## Contents

- [What we wanted to know](#what-we-wanted-to-know)
- [What we tested](#what-we-tested)
- [What happened](#what-happened)
- [What this means](#what-this-means)
- [Conditions for a useful future timing experiment](#conditions-for-a-useful-future-timing-experiment)
- [Limits](#limits)
- [Technical record](#technical-record)

## What we wanted to know

The scene comparison gave only a provisional model preference. The later work cared about serves, so we needed to compare the models on the same reviewed rally starts.

We also kept model choice separate from task success. A model can beat another model and still be unusable for the requested task. That is what happened here.

## What we tested

Qwen and Intern saw the same 32 independently reviewed rally starts. The human review found:

- 19 serves where physical contact was visible;
- 8 where contact happened off-frame;
- 4 where the broadcast omitted the serve;
- 1 unclear case.

The server was on the bottom side in 20 cases and the top side in 12.

Each model had to say who served, whether the contact was visible, and—only when visible—the exact contact frame.

All 19 visible contacts were inside the supplied clips, so a timing miss cannot be explained by the contact falling outside the video.

## What happened

![Reviewed rally-start comparison](../figures/clean_serve_gate.png)

| Measure | Intern | Qwen |
|---|---:|---:|
| Server correct | **23/32** | 14/32 |
| Serve visibility label correct | 19/32 | 19/32 |
| Visible contact close enough to the reviewed frame | 1/19 | 1/19 |
| Exact frame claimed when contact was not visible | 13/13 | 13/13 |

The server result clearly separated the models. Intern was the only model correct on 12 cases; Qwen was the only model correct on three. Qwen also answered `top` in 28 of 32 cases.

The 19/32 visibility score looks better than it was. **Both models answered `visible` in all 32 cases.** There happened to be 19 visible serves, so both models got those 19 right by always choosing the same label. They missed every off-frame, omitted, and unclear case.

Exact timing failed more seriously. Each model was close enough on only 1 of the 19 visible contacts, and both still supplied an exact frame in all 13 cases where physical contact was not visible.

## What this means

Intern is the better first model for any later VLM experiment that is otherwise justified.

That relative model choice is not evidence that Intern can reconstruct serves. From this interface, server side is the only field with enough signal to justify further study; serve visibility and exact contact timing are not.

The models were also unwilling to abstain even though the prompt allowed them to say `unclear` or return no contact frame. That behaviour means optional “I don’t know” fields are not a reliable abstention mechanism for a production system.

## Conditions for a useful future timing experiment

A useful later exact-contact experiment requires an independent way to verify the visual event, plus controls that reveal whether the model is following a marker or another stable input feature rather than tracking racket–shuttle contact.

One suitable control is a matched test that moves or removes a candidate marker and measures whether the answer follows it.

## Limits

This is a 32-case diagnostic, and 26 cases come from `sset_21`. The clips were built around existing automatic contact/camera-cut evidence rather than by searching whole matches from scratch. A marked cut in the video may also have influenced the answers.

## Technical record

The compact result is
[`2_final_model_gate.json.gz`](evidence/2_final_model_gate.json.gz). Exact
manifests, separate human truth, row-level scores, builder/scorer code, and
parser details are indexed in [`technical_index.md`](technical_index.md).
