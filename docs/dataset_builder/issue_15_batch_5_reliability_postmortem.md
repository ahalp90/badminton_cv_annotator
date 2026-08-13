# Issue 15 Batch 5 reliability postmortem

Status: accepted final trial, with follow-up hardening identified

This is the operational companion to the accepted [end-to-end trial
report](issue_15_batch_5_e2e_report.md). It separates observed facts from
inferences, records what was retained after each interruption, and describes
the recovery path an operator should use. It intentionally contains no
credentials, request payloads, or sensitive remote-host details.

## Executive summary

The final Issue 15 run completed the visual dataset path for two selected
videos: 218 rallies were assembled, and the final no-op resume preserved all
45 stage artifacts and four publication files byte-for-byte. The run was
therefore accepted for the visual dataset goal.

The run was not commentary-enriched. Gemini triage became unavailable after a
provider 503, so the reviewed visual fallback selected both videos and the
optional commentary path produced zero pairs. That is an intentional
availability policy, not evidence that commentary succeeded.

The trial exposed four different classes of failure:

| Class | Effect | Final disposition |
| --- | --- | --- |
| Provider credential readiness | Initial remote invocation could not authenticate | Corrected before expensive execution |
| Provider quota and availability | Commentary triage/cleaning could not complete | Bounded, recorded as unavailable, and visual processing continued |
| Deterministic code boundary | Exact EOF and isolated-child import prevented a visual run | Fixed and covered by focused regressions |
| Transient worker failure | Three pose shards failed during the first final run | Recovered by a dependency-lane resume; root cause was not retained |

## Timeline and evidence

| Attempt | What happened | What was preserved | Recovery |
| --- | --- | --- | --- |
| Initial remote preflight | The supplied provider credential was rejected as an unsupported access-token type. | No expensive pipeline work was accepted as reusable. | Correct credential/configuration before rerunning. |
| Early full trial | Commentary encountered a daily request quota, then provider high-demand failures left a request pending without a finite request deadline. The operator stopped it cleanly. | Completed manifest stages and their checksums remained available. | Batch 5C added a 120-second request bound, three application attempts, and terminal daily-quota classification. |
| First corrected visual run | A 153,600-frame source exposed an empty TrackNet EOF window; the isolated RTMLib child also imported a coordinator-only dependency absent from its production environment. | Failed run evidence and focused external gates were retained. | Batch 5D fixed both boundaries and added regression tests. |
| First final two-video run | One video completed; three of eight pose shards for the other exited nonzero. | The successful video and all upstream artifacts were reusable; failed pose and its dependants were invalidated. | A normal resume reran only the failed video’s pose dependency lane and downstream stages. |
| Repair verification | The repair supervisor compared the repaired two-video publications with a one-video baseline and reported a mismatch even though the pipeline completed correctly. | The final two-video publications and manifest remained intact. | A separate no-op resume verified the correct invariant: identical publications and stage artifacts after an unchanged rerun. |

The accepted final state and its timing evidence are recorded in the
[end-to-end report](issue_15_batch_5_e2e_report.md).

## What failed and why

### Provider credential rejection

The first provider call returned an authentication error identifying the
credential as the wrong access-token type. This was a credential/configuration
problem, not a pipeline retry problem. File permissions and non-empty-value
checks can confirm that a secret is present, but cannot prove that the provider
will accept it.

This could have been avoided by a deliberately small preflight request made
before download or model work. Such a probe consumes provider quota, so it is
best exposed as an explicit operator gate rather than silently issued by every
run.

### Quota, high demand, and unbounded requests

The next trial hit the provider’s daily request quota and later received a
provider 503/high-demand response. At that point the synchronous request did
not have a finite request deadline, so it could remain pending until the
operator intervened. The operator stopped the process cleanly.

Batch 5C changed this boundary so a request has a 120-second timeout, with at
most three application-level attempts and short backoff. A structured daily
quota response is terminal for that commentary batch: it is not retried for
later videos, avoiding waste of the remaining request budget. The resulting
optional stage is recorded as `unavailable`; it does not block the visual lane.

This behavior was confirmed again in the accepted run: a provider 503 made
triage unavailable, visual fallback proceeded, and no commentary pairs were
claimed.

### Exact frame-boundary and child-environment faults

The initial corrected run used a 153,600-frame video. Because that count is
exactly divisible by the TrackNet window/stride boundary, the iterable tried to
create a final window with no buffered frames and indexed an empty list. The
RTMLib child process also imported a module which required `frozendict`, even
though that package is intentionally not installed in the isolated RTMLib
environment.

Both were deterministic implementation faults. The fixes stop iteration on an
empty EOF buffer while retaining the existing padded partial-window path, and
move the coordinator-only dependency to the parent wrapper. The exact-multiple
TrackNet and isolated RTMLib-import gates passed in the final execution
environment.

### Failed pose shards in the first final run

During the first final two-video run, three pose shards for one video exited
nonzero while the other video’s pose stage succeeded. The subsequent rerun of
the same source, ranges, and configuration succeeded. That makes transient
startup, model-cache, or resource contention plausible, but **the exact cause
is unknown**: the sharding runner retained exit status but did not persist each
child’s stdout/stderr or host/GPU diagnostics.

This was recoverable because stage artifacts are validated and manifest-bound.
The failed pose stage and its dependants were invalidated; upstream downloaded
video, metadata, TrackNet proxy, and shuttle artifacts were reused. The
corrective resume reran only the required pose lane and downstream court,
annotation, pairing, projection, assembly, and report stages for that video.

## Could these failures have been avoided?

| Failure | Avoidable before the run? | Present protection | Recommended additional protection |
| --- | --- | --- | --- |
| Wrong provider credential | Yes, with an explicit live readiness probe | Configuration/file checks and clear failure | Add an opt-in one-request credential/quota preflight. |
| Daily quota / provider 503 | Not reliably; external-provider condition | Timeout, bounded retries, quota classification, visual fallback | Add a `--require-commentary` policy for runs where commentary is mandatory. |
| TrackNet exact EOF | Yes, through boundary tests | Fixed regression plus external exact-multiple gate | Keep frame-boundary cases in the normal suite. |
| RTMLib dependency split | Yes, through production-environment import gate | Fixed regression plus isolated child-import gate | Keep child import checks independent of coordinator extras. |
| Transient pose-shard exit | Not always | Safe stage invalidation and operator resume | Persist child diagnostics, warm model caches, and retry only failed validated shards. |
| Repair wrapper false alarm | Yes, by distinguishing repair from identity verification | Final no-op resume verification | Provide separate `repair-resume` and `verify-resume` supervisor modes. |

## Current recovery behavior

The pipeline is restartable, but it intentionally has no unattended
background retry daemon. An operator starts a normal `dataset_builder run` to
resume after a failure; already-valid work is not recomputed merely because the
command is invoked again.

| Situation | What is saved | What happens on a normal resume |
| --- | --- | --- |
| Completed, validated stage | Manifest record, outputs, checksums, semantic-validation evidence | Reused if the input fingerprint and outputs still validate. |
| Required-stage failure | Failure outcome and diagnostic report state; dependent stages invalidated | The affected stage and dependants rerun. Other videos and independent upstream stages may reuse. |
| Interrupted/crashed stage | Only atomically published completed artifacts are trusted | Incomplete output is not reusable; the stage is rerun after dependent invalidation. |
| Optional commentary unavailable | Unavailable outcome and integrity-checked status/chunk artifacts | Reused by default, so a quota-limited provider is not called again. |
| Optional commentary should be tried again | Same saved unavailable state | Use `--retry-unavailable` when the provider is known to have recovered. |
| Invalid/corrupt/missing output | Manifest entry is insufficient on its own | Validation fails, then the stage and dependants are invalidated and rebuilt. |

Publication files are not treated as success by themselves. The final
acceptance check is an unchanged resume with byte-identical publications and
stage artifacts. A repair resume is expected to change those files when it
finishes missing work, so it must not use the no-op identity assertion.

## Follow-up hardening

These are recommendations, not claims about the accepted implementation:

1. Add an explicit provider readiness/quota probe and a strict
   `--require-commentary` execution policy for commentary-dependent datasets.
2. Persist per-shard stdout/stderr, failure category, retry count, and basic
   host/GPU diagnostics in the pose-stage evidence.
3. Warm model/cache state in the exact child environment before launching a
   concurrent full run; if transient failures recur, add a configurable
   worker stagger or concurrency cap.
4. Retain validated individual pose shard artifacts and retry only failed
   shards with bounded backoff. Stitch only after every shard validates.
5. Split external trial helpers into a repair mode, which permits expected
   publication changes, and a verification mode, which requires exact
   no-op-resume identity.

The first two additions would make the next incident materially easier to
diagnose without weakening the current all-or-nothing publication guarantee.
