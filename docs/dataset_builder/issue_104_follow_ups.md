# Issue 104 follow-up gates

This document records what would be needed to revisit the Run 1 cut and
unresolved decisions. It is an unblock checklist, not a new experiment plan.
Issue #18 can implement the retained fields without waiting for these items.

Do not resume issue #16 from this list. Run the issue #104 benchmark again only
after a newer merged production change replaces an affected input, or after a
missing feature definition is resolved. Reuse the existing scorer and pinned
primitive artifacts where they remain compatible.

## Cut features

| Feature | Current evidence | Gate for reconsideration |
|---|---|---|
| Shots per rally | Exact production count on 298 of 3,287 eligible ShuttleSet rallies | A merged production contact stream with materially better complete-rally contact count and ordering, followed by the same aggregate and per-video ground-truth benchmark |
| Away-from-centre recovery | Formula and coverage are verified, but production contact and server attribution are weak | Better production contacts and server or striker attribution, followed by a comparison against windows built from existing human contact labels |
| Movement inefficiency | Formula and coverage are verified, but production intervals use predicted contacts | Better complete-rally contact sequences, followed by a paired comparison of predicted-contact and human-contact interval values on uniquely mapped rallies |

The existing ShuttleSet annotations are sufficient for those comparisons. They
do not require new vision inference. New annotations are needed only if the
existing contact and player labels prove unusable for a specific comparison.

## Unresolved features

| Feature | Gate for resolution |
|---|---|
| Rally duration | Define the offset after the final contact, including its base-30 frame units |
| Player sex | Add an authoritative metadata source; do not infer it from names or video |
| Serve speed proxy | Define return, static, and viewport-exit endpoints and missing-shuttle handling; then validate those events on a small reviewed sample |
| Raw degradation slope | First retain the underlying features and establish stable player identity across rallies and sets |
| Tanh-normalized degradation | Define the normalization temperature after the raw slope population exists |
| Commentary fields | Provide timestamped transcripts and a supported cleaning and pairing path, then validate sentiment, concept, timing, and player association |
| Backward extrapolation | Define the permitted scene boundary, maximum range, and provenance; then audit a small set of non-standard-view starts |

## Why issue 103 had no commentary

The issue #103 production configuration at
`/scratch/cmarti/issue103_ad8da4f/external/shuttleset-full.toml` explicitly set
`commentary.enabled = false`. Its SHA-256 is
`6e2a15ea3c44c4bc3cf8b38c461cdfd55c359178b49854080521949c07e93b20`.
The supported fixed-source runtime also marked every ShuttleSet source as
commentary-ineligible and deliberately bypassed commentary triage and cleaning.
A valid `GEMINI_API_KEY` would therefore not have enabled commentary in that
run.

The available inputs also contain no commentary payload to recover. ShuttleSet's
tracked annotations have no commentary or transcript fields. All 43 MP4 files
in the issue #103 fixed-source directory have no audio stream. The 40 generated
pairing artifacts contain 3,527 rally rows and zero paired commentary chunks;
they are explicit empty placeholders rather than omitted text.

Earlier issue #15 trials did encounter a rejected Gemini credential, daily
quota, and provider availability failures. Those failures explain why the
commentary lane was not operationally dependable at that time. The preserved
evidence does not show that the missing key was the sole reason for the issue
#103 configuration. Issue #103 was scoped to fixed-source visual primitive
extraction, and commentary was intentionally outside that run path.

To revisit commentary, first pull timed English captions from the recorded
source URLs where available. For the remaining videos, download audio-only
sources and run WhisperX; the current local MP4 files cannot be transcribed.
Then reuse the existing cleaning and pairing contracts. Fixed-source mode needs
a small supported extension to accept the transcript sidecars and mark those
sources commentary-eligible. The current replay command cannot do this as-is:
it restores the empty commentary state and starts at annotation.

After that extension, commentary acquisition, cleaning, pairing, projection,
assembly, and reporting can run separately while reusing the existing vision
and annotation artifacts. A validated cleaning-provider credential is required
only if the Gemini cleaner is retained. Do not create a second commentary
pipeline for this benchmark.

## Recommended order

1. Implement the Run 1 keep fields and reliability notes in issue #18.
2. Resolve definition-only items with Ari before doing more extraction.
3. Repeat this benchmark after a relevant merged production change.
4. Treat commentary as separate follow-up work if it remains part of the v1
   schema.
