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
| Rally-to-commentary association | The post-rally join covers 1,822 of 6,229 eligible human-contact rallies and 77 of 3,434 eligible production spans. The issue #136 aligned rerun word-times 92.6% of chunks; Ari's issue #138 rule then covers 2,119 of 6,156 human-contact rallies at 8 s and 2,222 at 10 s with no ambiguous chunk. 1,890 ShuttleSet chunk starts sit on replay-masked frames | Choose the lag window and a replay-time commentary policy with Ari, then label a sample that measures timing and rally-association accuracy. Aligned timestamps are done; see the [aligned rerun](issue_104_shuttleset_benchmark.md#aligned-rerun-issue-136) |

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
| Commentary sentiment, concept, and player link | Define supported output schemas, then validate each field against human labels before adding it to the dataset |
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

The commentary data preparation and benchmark are now complete. The exact 40
ShuttleSet and 47 non-overlap ShuttleSet22 sources have normalized transcripts.
Caption-backed sources use the existing timed-caption parser. The other 22 use
the existing coarse WhisperX path. Issue #136 then ran a whole-video WhisperX
pass with forced alignment over all 87 sources and re-timed 92.6% of the
cleaned chunks to word boundaries; the coarse transcripts stay in place as the
triage and cleaning inputs. Relevance triage and cleaning reuse the
supported provider contracts through OpenRouter support pinned at merge
`819d3075e72966a3d80eb454202b83b3810225ae`; native Gemini support is unchanged.

The final benchmark reuses the supported post-rally pairing contract and the
pinned issue #103 vision artifacts. It does not add a second commentary
pipeline or rerun vision inference. The [final commentary benchmark](issue_104_shuttleset_benchmark.md#final-commentary-benchmark)
records the evidence and decisions. Source text and segment timestamps can be
distributed alongside the visual and annotation data as an auxiliary,
source-aligned component for MLLM/VLM use and future research. Rally
association is cut from the authoritative v1 schema, while sentiment, concept,
and player link remain unresolved.

## Recommended order

1. Implement the Run 1 keep fields and reliability notes in issue #18.
2. Resolve definition-only items with Ari before doing more extraction.
3. Repeat this benchmark after a relevant merged production change.
4. Include the source-aligned commentary bundle as auxiliary supporting data.
   Keep raw, normalized, and cleaned artifacts separate, and revisit derived
   commentary fields only after their gates above are met.
5. For the rally link, settle the lag window and replay-time policy with Ari
   from the issue #136 lag sweep, then label a small sample of pairs before
   any accuracy claim.
