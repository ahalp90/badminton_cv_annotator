# Follow-up 4: compact evidence for serve reconstruction

## Bottom line

Neither enhanced InternVideo3 prompt passed the predeclared gate. Keep Intern
as the operational VLM on the clean Follow-up 2 interface. Do not widen this
serve route or run the conditional Qwen comparison.

Plain automatic observations appeared to improve contact timing from 1 to 9
of 19 visible contacts within project tolerance. That number does not show
that Intern found contact from the video. Intern repeated the supplied
inspection-point frame in 30 of the 31 replies that produced a frame. One
reply also failed the frozen parser because it wrapped valid JSON in a
Markdown fence.

Adding the current pipeline's server and contact proposals was worse. Server
correctness fell from 23 to 18 of 32 cases. This arm also missed the gate.

## What ran

Intern saw the same 32 clips retained from Follow-up 2. Clip hashes were
checked against the frozen manifest. Human serve labels remained separate
until scoring.

The clean Follow-up 2 result was reused as the baseline. It was not rerun.
Two enhanced prompts then added automatic support:

1. **Observations:** the automatic inspection point and selected camera cut,
   plus one-second frame counts for a usable full-court view, two on-court
   players and shuttle visibility. Explicit wrist or player-proximity results
   were included only where measured.
2. **Observations and proposals:** the same observations, followed by the
   current pipeline's proposed server and contact frame. The prompt said that
   both proposals could be wrong.

The support was written in ordinary sentences. It contained no human truth,
internal confidence score, replay label, raw keypoint array or trajectory.
The support-manifest SHA-256 before compression is
`e3f4708dbf6080014ceaeaf82ab27b8eb683605048e0b08a2903016e913c6035`.

Both 32-case runs used `yanziang/InternVideo3-8B-Instruct` at revision
`c4602918b65225650d152db2850fe34e01d21fcd`. They consumed all 120 frames per
clip. The observations run finished in 2 minutes 56 seconds. The proposal run
finished in 2 minutes 58 seconds. Both exited with code 0 and left the GPU
empty.

## Result

| Measure | Clean baseline | Observations | Observations + proposals |
|---|---:|---:|---:|
| Replies accepted by the frozen parser | 32 / 32 | 31 / 32 | 32 / 32 |
| Server correct | **23 / 32** | **23 / 32** | 18 / 32 |
| Serve state correct | 19 / 32 | 18 / 32 | **20 / 32** |
| Visible contact exact | 0 / 19 | **3 / 19** | 0 / 19 |
| Visible contact within project tolerance | 1 / 19 | **9 / 19** | 2 / 19 |
| Visible contact within ±10 base-30 frames | 1 / 19 | **17 / 19** | 3 / 19 |
| Mean contact error when a visible case had a frame | 46.8 frames | **5.5 frames** | 27.8 frames |
| Exact frame claimed without visible contact | 13 / 13 | 13 / 13 | **12 / 13** |

The observations arm did not improve serve state. It still called all 13
non-visible cases `visible` and supplied a frame for each one. Its one invalid
reply was a bare object wrapped in a Markdown fence, despite the prompt asking
for unfenced JSON. The retained parser did not repair that new error after the
result was known.

The timing columns are also dominated by the supplied prior. The support named
clip frame 40 as the inspection point in 26 cases and frame 80 in six. Intern
returned the named frame in 30 of the 31 parsed replies. The only other frame
was 108. The automatic clip builder deliberately placed those inspection
points near existing contact or cut evidence. Repeating them can therefore be
close to reviewed contact without showing that the model located racket–shuttle
contact.

The proposal arm did not simply inherit a useful answer. It gained one serve
state case and reduced unsupported frame claims by one, but lost five correct
server labels. Its contact timing remained poor.

## Decision

Run no Qwen confirmation. The observations arm was not parse-complete. Its
timing gain was also explained by copying the supplied inspection point. The
proposal arm breached the allowed two-case server drop.

Do not widen server attribution across the three fixtures and do not run the
end-to-end rally variants. Neither enhanced interface was genuinely promising
on the 32 reviewed starts.

Intern remains the operational choice from the clean Follow-up 2 comparison.
This result does not revise that completed finding. It shows that these compact
priors did not produce a dependable enhanced serve interface.

## Limits

This is a paired prompt comparison on 32 reviewed rally starts. Twenty-six
cases come from `sset_21`. The result does not cover unconstrained search
through a whole match.

The observations changed several sentences together. Their effects cannot be
separated. In particular, this run does not tell us whether court, player or
shuttle counts help when the exact inspection point is absent.

The first remote launch stopped before model loading because the runner depended
on a local helper package that was absent from the container. The replacement
runner loaded the same frozen support and clips. The failed launch produced no
model reply and did not contribute to scoring.

## Evidence

- [`../../experiments/rally_start_support_trials.py`](../../experiments/rally_start_support_trials.py)
  builds the truth-free support, runs both prompt arms and scores the retained
  replies.
- [`../../experiments/rally_start_support_remote.sh`](../../experiments/rally_start_support_remote.sh)
  contains the remote run boundary and GPU-release check.
- [`evidence/4_serve_support.json.gz`](evidence/4_serve_support.json.gz) contains
  the exact support sentences, source hashes and prompt hashes for all 32
  cases.
- [`evidence/4_serve_intern_scores.json.gz`](evidence/4_serve_intern_scores.json.gz)
  contains both row-level scores.
- [`evidence/4_serve_intern_remote_runs.tar.gz`](evidence/4_serve_intern_remote_runs.tar.gz)
  contains all 64 raw replies, exact prompts, model identities, sampling
  records, run statuses and post-run GPU checks. Remote paths were replaced
  with portable clip labels. Raw model text was not changed.
- [`4_serve_reconstruction.json.gz`](4_serve_reconstruction.json.gz) contains
  the main result, gate outcome and decision in machine-readable form.

The compressed support file has SHA-256
`2e80aebdebfde598cd7cad7a1f69c6b7cdb9d3b7a93f946a3162c4557429bed4`.
The score file has SHA-256
`db7b044cd9433175ab5ca4abca8dace70ae67049e769f8115ea0ce06e42d925a`.
The raw-run archive has SHA-256
`eda9b3c36f20264204856ae2cc44ff2832d48852f08497013d2c43139834bc40`.
