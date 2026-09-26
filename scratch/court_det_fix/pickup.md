# Court detector: resume here

Updated 27 September 2026. The detector works on prepared test images.
It still needs to run much faster and build its inputs from a new video.
Nothing is running. The next session is for GPU work, parallel work and Numba,
as the project owner requested on 26 September.

## Where work stopped

The separate research steps now run together in `court_detector/`. The detector
searches using all detected line fragments, then using only fragments that
look like court paint. It also builds courts from crossing-line templates.
Those searches are called G0 and G1 in the code. Both remain useful.

The default now rejects courts that require a sideways or upside-down camera.
When choosing a court, it combines paint and line support with a small net-post
bonus. It then adjusts the fit to the painted stripe edges or centres.
The [detector guide](court_detector/README.md) owns the API and settings;
[D19–D23](DETECTOR_DECISIONS.md#d19) record the checks and decisions.

The last detector session rejected a player-size filter and two further ways of
combining scores. The right court can still lose to one that slips a line at
the far end. [D23](DETECTOR_DECISIONS.md#d23) records these failures so they do
not reappear as untested ideas.

## Next session

Start with [the speed-up design](court_detector/PERFORMANCE.md). It preserves
the agreed constraints and the useful parts of the former handovers.

- Profile the current detector before choosing where to add GPU or compiled
  code. Older timing estimates predate the camera filter
- Run independent search pairs and court-scoring tasks in parallel, keeping
  their original order when collecting results
- Compare GPU and Numba approaches without maintaining two copies of the
  scoring maths. The backend choice and acceptable numerical differences
  still need measurement
- The agreed trial of cheap scores before full scores, and the agreed plan
  to reuse courts across scenes, remain open. Their designs are in the same
  document; this does not prescribe their order ahead of the next session

No new user ruling blocks reading, profiling or preparing that work. The
backend choice is still open. The old handover's suggested order is not a
new approval or a reason to repeat completed experiments.

## Limits to carry forward

The checked set has 20 court views and eight views without courts. Two of the
latter still receive false courts. One known far-end slip remains after the
kept scoring change. The dated [check reports](FP_INDEX.md#checks-and-measurements)
own the measurements; they do not establish performance on unseen cameras.

The detector needs a caller that supplies video frames, detected lines,
people, poses and scene ranges. Its search, scoring and geometry code now
lives in `court_detector/`; the old implementations are archived. Reusing a
court before searching a new scene is also unfinished; the existing annotator groups views after searching.

Real dark court markings have not established the stripe correction's
reliability. The earlier colour trials are closed. Better rejection of
non-court views and the frequency of serious fit errors remain open problems.
These limits do not reopen the old colour, net-weight or search-depth sweeps.

## Working state

The checkout was on `fix/court-det` at `c57d9a6e` when this tidy began.
`exp/court-det-opt2` was the earlier proposed branch for more speed-up work;
inspect the branches before resuming. This note grants no commit or push
permission. Check `git status` and `git log -1`; do not reset the checkout.

No experiment or remote session is active, as confirmed by the project owner.
Read `~/.codex/remote_hpc.md` before remote work. Some reruns need data held only
locally or on Carmack; [the file map](FP_INDEX.md#data-that-is-not-in-git)
identifies it. Historical launch instructions do not mean a job is still live.

[INDEX.md](INDEX.md) defines the document roles.
[The archive map](archive/README.md) leads to earlier records and recovery.
