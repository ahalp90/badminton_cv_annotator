# Issue 104 follow-up gates

This document records what would be needed to revisit the Run 1 cut and
unresolved decisions. It is an unblock checklist, not a new experiment plan.
Issue #18 can implement the retained fields without waiting for these items.

Do not resume issue #16 from this list. Run the issue #104 benchmark again only
after a newer merged production change replaces an affected input, or after a
missing feature definition is resolved. Reuse the existing scorer and pinned
primitive artifacts where they remain compatible.

## Cut features

No feature is currently cut. Issue #142 promoted shots per rally,
away-from-centre recovery, and movement inefficiency to keep once production
moved onto human ShuttleSet contacts. Issue #138 moved rally-to-commentary
association to unresolved once its lag rule shipped; see below.

## Unresolved features

| Feature | Gate for resolution |
|---|---|
| Rally duration | Define the offset after the final contact, including its base-30 frame units |
| Player sex | Add an authoritative metadata source; do not infer it from names or video |
| Serve speed proxy | Define return, static, and viewport-exit endpoints and missing-shuttle handling; then validate those events on a small reviewed sample |
| Commentary sentiment, concept, and player link | Define supported output schemas, then validate each field against human labels before adding it to the dataset |
| Backward extrapolation | Define the permitted scene boundary, maximum range, and provenance; then audit a small set of non-standard-view starts |
| Rally-to-commentary association | Coverage and construction are shipped in `commentary_rally_links` (Ari's issue #138 rule, 10 s lag, chunk-vs-two-rallies marked ambiguous, replay-masked starts flagged not dropped). Label a sample that measures whether a linked chunk actually discusses its rally, then move the disposition from unresolved to keep or narrow it |

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
association ships in `commentary_rally_links` at unresolved, not cut, because
issue #138 fixed its lag window and replay-time policy; sentiment, concept,
and player link remain unresolved and absent.

## Recommended order

1. Implement the Run 1 keep fields and reliability notes in issue #18.
2. Resolve definition-only items with Ari before doing more extraction.
3. Repeat this benchmark after a relevant merged production change.
4. Include the source-aligned commentary bundle as auxiliary supporting data.
   Keep raw, normalized, and cleaned artifacts separate, and revisit derived
   commentary fields only after their gates above are met.
5. For the rally link, label a small sample of `commentary_rally_links` pairs
   to measure accuracy before treating any pair as verified. The lag window
   (10 s) and replay-time policy (flag, don't drop) are already settled and
   shipped.
